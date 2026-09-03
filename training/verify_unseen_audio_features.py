import os
import pandas as pd

BASE = r"D:\DeepFake-Learning"
EXP = os.path.join(BASE, "research", "experiments")

TRAIN_CSV = os.path.join(EXP, "fakeavceleb_audio_unseen_rtvc_train.csv")
TEST_CSV = os.path.join(EXP, "fakeavceleb_audio_unseen_rtvc_test.csv")

print("=" * 70)
print("FAST UNSEEN AUDIO FEATURE VERIFICATION")
print("=" * 70)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

found_features = set()

# Load all existing audio metadata ONCE
for split in ["train", "validation", "test"]:
    meta_path = os.path.join(
        EXP, f"fakeavceleb_audio_{split}.csv"
    )

    df = pd.read_csv(meta_path)

    if "actual_path" in df.columns and "feature_path" in df.columns:
        for _, row in df.iterrows():
            feature_path = str(row["feature_path"])

            if os.path.exists(feature_path):
                found_features.add(str(row["actual_path"]))

print(f"\nExisting feature mappings: {len(found_features)}")

train_found = train_df["actual_path"].astype(str).isin(found_features).sum()
test_found = test_df["actual_path"].astype(str).isin(found_features).sum()

print("\nUNSEEN TRAIN SET")
print(f"Samples          : {len(train_df)}")
print(f"Features found   : {train_found}")
print(f"Features missing : {len(train_df) - train_found}")

print("\nRTVC UNSEEN TEST")
print(f"Samples          : {len(test_df)}")
print(f"Features found   : {test_found}")
print(f"Features missing : {len(test_df) - test_found}")

print("\n" + "=" * 70)

if train_found == len(train_df) and test_found == len(test_df):
    print("PASS: ALL FEATURES ARE READY")
else:
    print("WARNING: SOME FEATURES ARE MISSING")

print("=" * 70)