import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_features"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "outputs"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 5e-4

INPUT_SIZE = 512
HIDDEN_SIZE = 128

DROPOUT = 0.3

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# DATASET
# =========================================================

class FeatureSequenceDataset(Dataset):

    def __init__(self, feature_file):

        data = torch.load(
            feature_file,
            map_location="cpu"
        )

        self.features = data["features"].float()
        self.labels = data["labels"].float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index]
        )


# =========================================================
# LOAD DATA
# =========================================================

train_dataset = FeatureSequenceDataset(
    FEATURE_DIR / "train_temporal_features.pt"
)

val_dataset = FeatureSequenceDataset(
    FEATURE_DIR / "validation_temporal_features.pt"
)

test_dataset = FeatureSequenceDataset(
    FEATURE_DIR / "test_temporal_features.pt"
)


# =========================================================
# INFORMATION
# =========================================================

class_counts = (
    train_dataset.labels
    .long()
    .bincount()
)

print("=" * 70)
print("BiLSTM + TEMPORAL ATTENTION TRAINING")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)

print()
print("Training class distribution:")

print(
    "Real:",
    int(class_counts[0])
)

print(
    "Fake:",
    int(class_counts[1])
)


# =========================================================
# WEIGHTED SAMPLER
# =========================================================

class_weights = {
    0: 1.0 / class_counts[0].item(),
    1: 1.0 / class_counts[1].item()
}

sample_weights = torch.tensor(
    [
        class_weights[int(label.item())]
        for label in train_dataset.labels
    ],
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =========================================================
# BiLSTM + TEMPORAL ATTENTION
# =========================================================

class BiLSTMTemporalAttention(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=128,
        dropout=0.3
    ):

        super().__init__()

        # -------------------------------------------------
        # Bidirectional LSTM
        # -------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM output dimension = hidden_size * 2

        attention_dim = hidden_size * 2

        # -------------------------------------------------
        # Temporal Attention
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                attention_dim,
                128
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                1
            )
        )


    def forward(self, x):

        # x:
        # [batch, 5, 512]

        # -------------------------------------------------
        # BiLSTM
        # -------------------------------------------------

        lstm_output, _ = self.lstm(x)

        # [batch, 5, 256]

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        attention_scores = self.attention(
            lstm_output
        )

        # [batch, 5, 1]

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # -------------------------------------------------
        # Weighted temporal representation
        # -------------------------------------------------

        temporal_features = (
            lstm_output
            * attention_weights
        ).sum(dim=1)

        # [batch, 256]

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        output = self.classifier(
            temporal_features
        )

        return output


model = BiLSTMTemporalAttention(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT
).to(device)


# =========================================================
# LOSS + OPTIMIZER
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# PARAMETERS
# =========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("Model parameters:")

print(
    "Total:",
    f"{total_params:,}"
)

print(
    "Trainable:",
    f"{trainable_params:,}"
)


# =========================================================
# CHECKPOINT
# =========================================================

best_auc = -1.0

best_checkpoint = (
    CHECKPOINT_DIR
    / "temporal_bilstm_attention_best.pth"
)


# =========================================================
# TRAINING
# =========================================================

for epoch in range(EPOCHS):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_labels = []

    for features, labels in train_loader:

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            features
        ).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * features.size(0)
        )

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss_total = 0.0

    val_probabilities = []
    val_labels = []

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                features
            ).squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * features.size(0)
            )

            probabilities = torch.sigmoid(
                outputs
            )

            val_probabilities.extend(
                probabilities.cpu()
                .numpy()
            )

            val_labels.extend(
                labels.cpu()
                .numpy()
            )


    val_loss = (
        val_loss_total
        / len(val_dataset)
    )

    val_labels_array = np.array(
        val_labels
    )

    val_probabilities_array = np.array(
        val_probabilities
    )

    val_predictions = (
        val_probabilities_array >= 0.5
    ).astype(int)


    val_accuracy = accuracy_score(
        val_labels_array,
        val_predictions
    )

    val_balanced_accuracy = (
        balanced_accuracy_score(
            val_labels_array,
            val_predictions
        )
    )

    val_auc = roc_auc_score(
        val_labels_array,
        val_probabilities_array
    )


    # =====================================================
    # PRINT
    # =====================================================

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: "
        f"{val_accuracy * 100:.2f}% | "
        f"Val Balanced Acc: "
        f"{val_balanced_accuracy * 100:.2f}% | "
        f"Val ROC-AUC: "
        f"{val_auc:.4f}"
    )


    # =====================================================
    # SAVE BEST CHECKPOINT
    # =====================================================

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_auc":
                    val_auc,

                "val_accuracy":
                    val_accuracy,

                "val_balanced_accuracy":
                    val_balanced_accuracy
            },
            best_checkpoint
        )

        print(
            "✓ Best checkpoint saved"
        )


# =========================================================
# LOAD BEST MODEL
# =========================================================

print()
print("=" * 70)
print("LOADING BEST BiLSTM + ATTENTION")
print("=" * 70)

checkpoint = torch.load(
    best_checkpoint,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Best Epoch:",
    checkpoint["epoch"]
)

print(
    "Best Validation ROC-AUC:",
    f"{checkpoint['val_auc']:.4f}"
)


# =========================================================
# TEST EVALUATION
# =========================================================

print()
print("=" * 70)
print("BiLSTM + TEMPORAL ATTENTION TEST EVALUATION")
print("=" * 70)

test_probabilities = []
test_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(
            features
        ).squeeze(1)

        probabilities = torch.sigmoid(
            outputs
        )

        test_probabilities.extend(
            probabilities.cpu()
            .numpy()
        )

        test_labels.extend(
            labels.cpu()
            .numpy()
        )


# =========================================================
# TEST METRICS
# =========================================================

test_labels_array = np.array(
    test_labels
)

test_probabilities_array = np.array(
    test_probabilities
)

test_predictions = (
    test_probabilities_array >= 0.5
).astype(int)


test_accuracy = accuracy_score(
    test_labels_array,
    test_predictions
)

test_balanced_accuracy = (
    balanced_accuracy_score(
        test_labels_array,
        test_predictions
    )
)

test_auc = roc_auc_score(
    test_labels_array,
    test_probabilities_array
)

test_precision = precision_score(
    test_labels_array,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_labels_array,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels_array,
    test_predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels_array,
    test_predictions
)


# =========================================================
# RESULTS
# =========================================================

print()

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Balanced Accuracy: "
    f"{test_balanced_accuracy * 100:.2f}%"
)

print(
    f"Test ROC-AUC: "
    f"{test_auc:.4f}"
)

print(
    f"Test Precision: "
    f"{test_precision:.4f}"
)

print(
    f"Test Recall: "
    f"{test_recall:.4f}"
)

print(
    f"Test F1 Score: "
    f"{test_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 70)
print("BiLSTM + TEMPORAL ATTENTION COMPLETE")
print("=" * 70)

print(
    "Best Validation ROC-AUC:",
    f"{best_auc:.4f}"
)

print(
    "Test ROC-AUC:",
    f"{test_auc:.4f}"
)

print()
print(
    "Checkpoint:",
    best_checkpoint
)

print("=" * 70)