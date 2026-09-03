from fastapi import FastAPI, UploadFile, File

import os
import shutil
import subprocess

import torch
import torch.nn as nn
import cv2
import numpy as np
import librosa

from PIL import Image
from torchvision import transforms

from models.resnet_baseline import ResNetBaseline
from models.audio_cnn import AudioCNN
from models.fusion_baseline import FusionBaseline


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Deepfake Detection & Learning System",
    version="1.0.0"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\DeepFake-Learning"

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "app",
    "backend",
    "uploads"
)

VISUAL_CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "resnet18_face_crop_best.pth"
)

TEMPORAL_CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "temporal_lstm_best.pth"
)

BILSTM_ATTENTION_CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "temporal_bilstm_attention_best.pth"
)

AUDIO_CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "audio_cnn_best.pth"
)
FUSION_CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "fusion_best.pth"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Inference device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    cascade_path
)


# ============================================================
# VISUAL RESNET18
# ============================================================

print()
print("=" * 60)
print("LOADING VISUAL MODEL")
print("=" * 60)

visual_model = ResNetBaseline().to(device)

visual_checkpoint = torch.load(
    VISUAL_CHECKPOINT,
    map_location=device
)

if (
    isinstance(visual_checkpoint, dict)
    and "model_state_dict" in visual_checkpoint
):

    visual_state_dict = (
        visual_checkpoint["model_state_dict"]
    )

elif (
    isinstance(visual_checkpoint, dict)
    and "state_dict" in visual_checkpoint
):

    visual_state_dict = (
        visual_checkpoint["state_dict"]
    )

else:

    visual_state_dict = visual_checkpoint


visual_model.load_state_dict(
    visual_state_dict
)

visual_model.eval()

print(
    "Visual model loaded successfully."
)


# ============================================================
# TEMPORAL LSTM
# ============================================================

class TemporalLSTM(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=256,
        dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        temporal_features = (
            lstm_output[:, -1, :]
        )

        output = self.classifier(
            temporal_features
        )

        return output


print()
print("=" * 60)
print("LOADING TEMPORAL LSTM")
print("=" * 60)

temporal_model = TemporalLSTM(
    input_size=512,
    hidden_size=256,
    dropout=0.3
).to(device)

temporal_checkpoint = torch.load(
    TEMPORAL_CHECKPOINT,
    map_location=device
)

if (
    isinstance(temporal_checkpoint, dict)
    and "model_state_dict" in temporal_checkpoint
):

    temporal_state_dict = (
        temporal_checkpoint["model_state_dict"]
    )

else:

    temporal_state_dict = temporal_checkpoint


temporal_model.load_state_dict(
    temporal_state_dict
)

temporal_model.eval()

print(
    "Temporal LSTM loaded successfully."
)

if isinstance(
    temporal_checkpoint,
    dict
):

    if "epoch" in temporal_checkpoint:

        print(
            "Best Epoch:",
            temporal_checkpoint["epoch"]
        )

    if "val_auc" in temporal_checkpoint:

        print(
            "Validation ROC-AUC:",
            f"{temporal_checkpoint['val_auc']:.4f}"
        )


# ============================================================
# BiLSTM + TEMPORAL ATTENTION
# ============================================================

class BiLSTMTemporalAttention(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=128,
        dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        attention_dim = (
            hidden_size * 2
        )

        self.attention = nn.Sequential(

            nn.Linear(
                attention_dim,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                attention_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        attention_scores = self.attention(
            lstm_output
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        temporal_features = (
            lstm_output
            * attention_weights
        ).sum(dim=1)

        output = self.classifier(
            temporal_features
        )

        return output


print()
print("=" * 60)
print("LOADING BiLSTM + TEMPORAL ATTENTION")
print("=" * 60)

bilstm_attention_model = BiLSTMTemporalAttention(
    input_size=512,
    hidden_size=128,
    dropout=0.3
).to(device)

bilstm_checkpoint = torch.load(
    BILSTM_ATTENTION_CHECKPOINT,
    map_location=device
)

if (
    isinstance(bilstm_checkpoint, dict)
    and "model_state_dict" in bilstm_checkpoint
):

    bilstm_state_dict = (
        bilstm_checkpoint["model_state_dict"]
    )

else:

    bilstm_state_dict = bilstm_checkpoint


bilstm_attention_model.load_state_dict(
    bilstm_state_dict
)

bilstm_attention_model.eval()

print(
    "BiLSTM + Attention loaded successfully."
)

if isinstance(
    bilstm_checkpoint,
    dict
):

    if "epoch" in bilstm_checkpoint:

        print(
            "Best Epoch:",
            bilstm_checkpoint["epoch"]
        )

    if "val_auc" in bilstm_checkpoint:

        print(
            "Validation ROC-AUC:",
            f"{bilstm_checkpoint['val_auc']:.4f}"
        )


# ============================================================
# AUDIO CNN
# ============================================================

print()
print("=" * 60)
print("LOADING AUDIO CNN")
print("=" * 60)

audio_model = AudioCNN().to(device)

audio_checkpoint = torch.load(
    AUDIO_CHECKPOINT,
    map_location=device
)

if (
    isinstance(audio_checkpoint, dict)
    and "model_state_dict" in audio_checkpoint
):

    audio_state_dict = (
        audio_checkpoint["model_state_dict"]
    )

elif (
    isinstance(audio_checkpoint, dict)
    and "state_dict" in audio_checkpoint
):

    audio_state_dict = (
        audio_checkpoint["state_dict"]
    )

else:

    audio_state_dict = audio_checkpoint


audio_model.load_state_dict(
    audio_state_dict
)

audio_model.eval()

print(
    "Audio CNN loaded successfully."
)

if isinstance(
    audio_checkpoint,
    dict
):

    if "epoch" in audio_checkpoint:

        print(
            "Best Epoch:",
            audio_checkpoint["epoch"]
        )

    if "val_auc" in audio_checkpoint:

        print(
            "Validation ROC-AUC:",
            f"{audio_checkpoint['val_auc']:.4f}"
        )
        # ============================================================
# FUSION MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING FUSION MODEL")
print("=" * 60)

fusion_model = FusionBaseline().to(device)

fusion_checkpoint = torch.load(
    FUSION_CHECKPOINT,
    map_location=device
)

if (
    isinstance(fusion_checkpoint, dict)
    and "model_state_dict" in fusion_checkpoint
):

    fusion_state_dict = (
        fusion_checkpoint["model_state_dict"]
    )

elif (
    isinstance(fusion_checkpoint, dict)
    and "state_dict" in fusion_checkpoint
):

    fusion_state_dict = (
        fusion_checkpoint["state_dict"]
    )

else:

    fusion_state_dict = fusion_checkpoint


fusion_model.load_state_dict(
    fusion_state_dict
)

fusion_model.eval()

print(
    "Fusion model loaded successfully."
)

if isinstance(
    fusion_checkpoint,
    dict
):

    if "epoch" in fusion_checkpoint:

        print(
            "Best Epoch:",
            fusion_checkpoint["epoch"]
        )

    if "val_auc" in fusion_checkpoint:

        print(
            "Validation ROC-AUC:",
            f"{fusion_checkpoint['val_auc']:.4f}"
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "online",

        "message":
            "Deepfake Detection & Learning System"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "device":
            str(device),

        "visual_model":
            "loaded",

        "temporal_lstm":
            "loaded",

        "bilstm_attention":
            "loaded",

        "audio_model":
            "loaded",
            
        "fusion_model":
            "loaded"    

    }


# ============================================================
# ANALYZE / UPLOAD
# ============================================================

# ============================================================
# MULTIMODAL ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze_media(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        # ====================================================
        # VISUAL ANALYSIS
        # ====================================================

        cap = cv2.VideoCapture(
            file_path
        )

        if not cap.isOpened():

            return {
                "success": False,
                "error": "Could not open video"
            }

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0

        sequence_length = 5

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            sequence_length,
            dtype=int
        )

        feature_sequence = []
        visual_predictions = []

        frame_results = []

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                continue

            x, y, w, h = max(
                faces,
                key=lambda face:
                face[2] * face[3]
            )

            margin = int(
                0.15 * max(w, h)
            )

            x1 = max(
                0,
                x - margin
            )

            y1 = max(
                0,
                y - margin
            )

            x2 = min(
                frame.shape[1],
                x + w + margin
            )

            y2 = min(
                frame.shape[0],
                y + h + margin
            )

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = cv2.resize(
                face,
                (224, 224)
            )

            face_image = Image.fromarray(
                face
            )

            tensor = image_transform(
                face_image
            )

            tensor = tensor.unsqueeze(
                0
            ).to(device)

            with torch.no_grad():

                visual_logit = visual_model(
                    tensor
                )

                visual_probability = (
                    torch.sigmoid(
                        visual_logit
                    ).item()
                )

                features = (
                    visual_model.extract_features(
                        tensor
                    )
                )

            visual_predictions.append(
                visual_probability
            )

            feature_sequence.append(
                features.squeeze(0)
            )

            frame_results.append({
                "frame": int(frame_index),
                "fake_probability": round(
                    visual_probability,
                    4
                ),
                "face_detected": True
            })

        cap.release()

        if len(feature_sequence) == 0:

            return {
                "success": False,
                "error": "No face detected"
            }

        visual_probability = float(
            np.mean(
                visual_predictions
            )
        )

        # ====================================================
        # TEMPORAL ANALYSIS
        # ====================================================

        temporal_probability = None
        bilstm_probability = None

        if len(feature_sequence) == 5:

            temporal_input = torch.stack(
                feature_sequence
            ).unsqueeze(
                0
            ).to(device)

            with torch.no_grad():

                temporal_logit = (
                    temporal_model(
                        temporal_input
                    )
                )

                temporal_probability = (
                    torch.sigmoid(
                        temporal_logit
                    ).item()
                )

                bilstm_logit = (
                    bilstm_attention_model(
                        temporal_input
                    )
                )

                bilstm_probability = (
                    torch.sigmoid(
                        bilstm_logit
                    ).item()
                )

        # ====================================================
        # AUDIO ANALYSIS
        # ====================================================

        wav_path = os.path.join(
            UPLOAD_DIR,
            file.filename + "_audio.wav"
        )

        try:

            extract_audio_from_video(
                file_path,
                wav_path
            )

            audio_feature = (
                prepare_audio_feature(
                    wav_path
                )
            )

            audio_feature = (
                audio_feature.to(device)
            )

            with torch.no_grad():

                audio_logit = audio_model(
                    audio_feature
                )

                audio_probability = (
                    torch.sigmoid(
                        audio_logit
                    ).item()
                )

        finally:

            if os.path.exists(wav_path):

                os.remove(wav_path)

        # ====================================================
        # MULTIMODAL FUSION
        # ====================================================

        visual_tensor = torch.tensor(
            [[visual_probability]],
            dtype=torch.float32,
            device=device
        )

        audio_tensor = torch.tensor(
            [[audio_probability]],
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():

            fusion_logit = fusion_model(
                visual_tensor,
                audio_tensor
            )

            fusion_probability = (
                torch.sigmoid(
                    fusion_logit
                ).item()
            )

        # ====================================================
        # EVIDENCE GENERATION
        # ====================================================

        if (
            visual_probability >= 0.5
            and audio_probability >= 0.5
        ):

            evidence_type = (
                "cross-modal agreement"
            )

            explanation = (
                "Both visual and audio "
                "analysis indicate manipulation."
            )

        elif (
            visual_probability < 0.5
            and audio_probability < 0.5
        ):

            evidence_type = (
                "cross-modal real"
            )

            explanation = (
                "Both visual and audio "
                "analysis are below the "
                "manipulation threshold."
            )

        elif audio_probability >= 0.5:

            evidence_type = (
                "audio-dominant"
            )

            explanation = (
                "Audio analysis indicates "
                "manipulation while visual "
                "analysis is below threshold."
            )

        else:

            evidence_type = (
                "visual-dominant"
            )

            explanation = (
                "Visual analysis indicates "
                "manipulation while audio "
                "analysis is below threshold."
            )

        # ====================================================
        # FINAL DECISION
        # ====================================================

        prediction = (
            "FAKE"
            if fusion_probability >= 0.5
            else "REAL"
        )

        confidence = max(
            fusion_probability,
            1.0 - fusion_probability
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "prediction": prediction,

            "confidence": round(
                confidence,
                4
            ),

            "visual": {

                "fake_probability": round(
                    visual_probability,
                    4
                ),

                "real_probability": round(
                    1.0 - visual_probability,
                    4
                ),

                "frames": frame_results

            },

            "temporal_lstm": {

                "fake_probability": (
                    round(
                        temporal_probability,
                        4
                    )
                    if temporal_probability is not None
                    else None
                ),

                "real_probability": (
                    round(
                        1.0 - temporal_probability,
                        4
                    )
                    if temporal_probability is not None
                    else None
                )

            },

            "bilstm_attention": {

                "fake_probability": (
                    round(
                        bilstm_probability,
                        4
                    )
                    if bilstm_probability is not None
                    else None
                ),

                "real_probability": (
                    round(
                        1.0 - bilstm_probability,
                        4
                    )
                    if bilstm_probability is not None
                    else None
                )

            },

            "audio": {

                "fake_probability": round(
                    audio_probability,
                    4
                ),

                "real_probability": round(
                    1.0 - audio_probability,
                    4
                ),

                "sample_rate": 16000,

                "channels": 1,

                "feature_shape": [
                    64,
                    512
                ]

            },

            "fusion": {

                "fake_probability": round(
                    fusion_probability,
                    4
                ),

                "real_probability": round(
                    1.0 - fusion_probability,
                    4
                )

            },

            "evidence": {

                "type": evidence_type,

                "explanation": explanation

            },

            "video": {

                "total_frames": total_frames,

                "fps": round(
                    fps,
                    3
                ),

                "duration_seconds": round(
                    total_frames / fps,
                    3
                ),

                "processed_frames": len(
                    feature_sequence
                )

            },

            "device": str(device)

        }

    except Exception as e:

        print(
            "Analysis error:",
            str(e)
        )

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return {

            "success":
                False,

            "error":
                "Could not read image"

        }


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(80, 80)

    )


    if len(faces) == 0:

        return {

            "success":
                False,

            "error":
                "No face detected"

        }


    x, y, w, h = max(

        faces,

        key=lambda face:
            face[2] * face[3]

    )


    margin = int(
        0.15 * max(w, h)
    )


    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        image.shape[1],
        x + w + margin
    )

    y2 = min(
        image.shape[0],
        y + h + margin
    )


    face = image[
        y1:y2,
        x1:x2
    ]


    if face.size == 0:

        return {

            "success":
                False,

            "error":
                "Invalid face crop"

        }


    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    face = cv2.resize(
        face,
        (224, 224)
    )


    face_image = Image.fromarray(
        face
    )


    tensor = image_transform(
        face_image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)


    with torch.no_grad():

        logit = visual_model(
            tensor
        )

        fake_probability = (
            torch.sigmoid(
                logit
            ).item()
        )


    prediction = (

        "FAKE"

        if fake_probability >= 0.5

        else "REAL"

    )


    return {

        "success":
            True,

        "prediction":
            prediction,

        "fake_probability":
            round(
                fake_probability,
                4
            ),

        "real_probability":
            round(
                1 - fake_probability,
                4
            ),

        "face_detected":
            True,

        "face_count":
            len(faces),

        "device":
            str(device)

    }


# ============================================================
# VIDEO PREDICTION
# ============================================================

@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...)
):

    video_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        video_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        return {

            "success":
                False,

            "error":
                "Could not open video"

        }


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 25.0


    duration = (

        total_frames / fps

        if total_frames > 0

        else 0

    )


    sequence_length = 5


    if total_frames < sequence_length:

        cap.release()

        return {

            "success":
                False,

            "error":
                "Video must contain at least 5 frames"

        }


    frame_indices = np.linspace(

        0,

        total_frames - 1,

        sequence_length,

        dtype=int

    )


    feature_sequence = []

    frame_results = []


    for frame_index in frame_indices:

        cap.set(

            cv2.CAP_PROP_POS_FRAMES,

            int(frame_index)

        )


        success, frame = cap.read()


        if not success:

            continue


        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )


        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(80, 80)

        )


        if len(faces) == 0:

            frame_results.append({

                "frame":
                    int(frame_index),

                "timestamp":
                    round(
                        frame_index / fps,
                        3
                    ),

                "face_detected":
                    False,

                "fake_probability":
                    None

            })

            continue


        x, y, w, h = max(

            faces,

            key=lambda face:
                face[2] * face[3]

        )


        margin = int(

            0.15 * max(w, h)

        )


        x1 = max(
            0,
            x - margin
        )

        y1 = max(
            0,
            y - margin
        )

        x2 = min(
            frame.shape[1],
            x + w + margin
        )

        y2 = min(
            frame.shape[0],
            y + h + margin
        )


        face = frame[
            y1:y2,
            x1:x2
        ]


        if face.size == 0:

            continue


        face = cv2.cvtColor(

            face,

            cv2.COLOR_BGR2RGB

        )


        face = cv2.resize(

            face,

            (224, 224)

        )


        face_image = Image.fromarray(
            face
        )


        tensor = image_transform(
            face_image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(device)


        with torch.no_grad():

            features = (
                visual_model.extract_features(
                    tensor
                )
            )

            visual_logit = (
                visual_model(
                    tensor
                )
            )

            visual_probability = (
                torch.sigmoid(
                    visual_logit
                ).item()
            )


        feature_sequence.append(
            features.squeeze(0)
        )


        frame_results.append({

            "frame":
                int(frame_index),

            "timestamp":
                round(
                    frame_index / fps,
                    3
                ),

            "face_detected":
                True,

            "fake_probability":
                round(
                    visual_probability,
                    4
                ),

            "face_box": {

                "x":
                    int(x),

                "y":
                    int(y),

                "width":
                    int(w),

                "height":
                    int(h)

            }

        })


    cap.release()


    if len(feature_sequence) != 5:

        return {

            "success":
                False,

            "error":
                "Could not obtain 5 valid face features",

            "processed_frames":
                len(feature_sequence)

        }


    # ========================================================
    # TEMPORAL INPUT
    # [1, 5, 512]
    # ========================================================

    temporal_input = torch.stack(

        feature_sequence,

        dim=0

    ).unsqueeze(
        0
    ).to(device)


    # ========================================================
    # TEMPORAL LSTM
    # ========================================================

    with torch.no_grad():

        temporal_logit = (
            temporal_model(
                temporal_input
            )
        )

        temporal_fake_probability = (
            torch.sigmoid(
                temporal_logit
            ).item()
        )


    # ========================================================
    # BiLSTM + ATTENTION
    # ========================================================

    with torch.no_grad():

        bilstm_logit = (
            bilstm_attention_model(
                temporal_input
            )
        )

        bilstm_fake_probability = (
            torch.sigmoid(
                bilstm_logit
            ).item()
        )


    # ========================================================
    # VISUAL MEAN
    # ========================================================

    visual_predictions = [

        item["fake_probability"]

        for item in frame_results

        if item["fake_probability"]
        is not None

    ]


    visual_mean = float(

        np.mean(
            visual_predictions
        )

    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    temporal_prediction = (

        "FAKE"

        if temporal_fake_probability >= 0.5

        else "REAL"

    )


    bilstm_prediction = (

        "FAKE"

        if bilstm_fake_probability >= 0.5

        else "REAL"

    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "prediction":
            temporal_prediction,

        "visual_mean_fake_probability":
            round(
                visual_mean,
                4
            ),

        "temporal_lstm": {

            "prediction":
                temporal_prediction,

            "fake_probability":
                round(
                    temporal_fake_probability,
                    4
                ),

            "real_probability":
                round(
                    1 - temporal_fake_probability,
                    4
                )

        },

        "bilstm_attention": {

            "prediction":
                bilstm_prediction,

            "fake_probability":
                round(
                    bilstm_fake_probability,
                    4
                ),

            "real_probability":
                round(
                    1 - bilstm_fake_probability,
                    4
                )

        },

        "video": {

            "filename":
                file.filename,

            "total_frames":
                total_frames,

            "fps":
                round(
                    fps,
                    3
                ),

            "duration_seconds":
                round(
                    duration,
                    3
                ),

            "sampled_frames":
                5,

            "processed_frames":
                len(
                    feature_sequence
                )

        },

        "frame_results":
            frame_results,

        "device":
            str(device)

    }


# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def extract_audio_from_video(
    video_path,
    wav_path
):

    command = [

        "ffmpeg",

        "-y",

        "-i",
        video_path,

        "-vn",

        "-ac",
        "1",

        "-ar",
        "16000",

        "-f",
        "wav",

        wav_path

    ]


    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE

    )


    if result.returncode != 0:

        raise RuntimeError(

            result.stderr.decode(
                errors="ignore"
            )

        )


# ============================================================
# AUDIO FEATURE PREPARATION
# ============================================================

def prepare_audio_feature(
    wav_path
):

    waveform, sample_rate = librosa.load(

        wav_path,

        sr=16000,

        mono=True

    )


    # --------------------------------------------------------
    # Log-Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=waveform,

        sr=sample_rate,

        n_mels=64

    )


    log_mel = librosa.power_to_db(

        mel,

        ref=np.max

    )


    # --------------------------------------------------------
    # Exactly 512 time frames
    # --------------------------------------------------------

    target_frames = 512


    if log_mel.shape[1] < target_frames:

        padding = (

            target_frames
            - log_mel.shape[1]

        )


        log_mel = np.pad(

            log_mel,

            (
                (0, 0),
                (0, padding)
            ),

            mode="constant"

        )

    else:

        log_mel = (

            log_mel[
                :,
                :target_frames
            ]

        )


    # --------------------------------------------------------
    # Tensor
    # [1, 1, 64, 512]
    # --------------------------------------------------------

    feature = torch.tensor(

        log_mel,

        dtype=torch.float32

    )


    feature = feature.unsqueeze(
        0
    )

    feature = feature.unsqueeze(
        0
    )


    return feature


# ============================================================
# AUDIO PREDICTION
# ============================================================

@app.post("/predict-audio")
async def predict_audio(
    file: UploadFile = File(...)
):

    input_path = os.path.join(

        UPLOAD_DIR,

        file.filename

    )


    with open(

        input_path,

        "wb"

    ) as buffer:

        shutil.copyfileobj(

            file.file,

            buffer

        )


    wav_path = os.path.join(

        UPLOAD_DIR,

        file.filename
        + "_audio.wav"

    )


    try:

        # ----------------------------------------------------
        # Extract audio
        # ----------------------------------------------------

        extract_audio_from_video(

            input_path,

            wav_path

        )


        # ----------------------------------------------------
        # Prepare log-mel
        # ----------------------------------------------------

        audio_feature = (

            prepare_audio_feature(

                wav_path

            )

        )


        audio_feature = (

            audio_feature.to(device)

        )


        # ----------------------------------------------------
        # Audio CNN
        # ----------------------------------------------------

        with torch.no_grad():

            audio_logit = (

                audio_model(

                    audio_feature

                )

            )


            fake_probability = (

                torch.sigmoid(

                    audio_logit

                ).item()

            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = (

            "FAKE"

            if fake_probability >= 0.5

            else "REAL"

        )


        return {

            "success":
                True,

            "prediction":
                prediction,

            "fake_probability":
                round(
                    fake_probability,
                    4
                ),

            "real_probability":
                round(
                    1 - fake_probability,
                    4
                ),

            "audio": {

                "sample_rate":
                    16000,

                "channels":
                    1,

                "feature_shape":
                    [
                        64,
                        512
                    ]

            },

            "device":
                str(device)

        }


    except Exception as e:

        return {

            "success":
                False,

            "error":
                str(e)

        }


    finally:

        if os.path.exists(
            wav_path
        ):

            os.remove(
                wav_path
            )