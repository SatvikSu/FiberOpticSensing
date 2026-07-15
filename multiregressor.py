import os
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

start_time = time.time()

train_top_path = "/Users/samayjain/PycharmProjects/OpenAIVers/train_top_UNIQUE.csv"
train_side_path = "/Users/samayjain/PycharmProjects/OpenAIVers/train_side_UNIQUE.csv"
train_intensity_path = "/Users/samayjain/PycharmProjects/OpenAIVers/train_intensity_UNIQUE.csv"

test_top_path = "/Users/samayjain/PycharmProjects/OpenAIVers/test_top_UNIQUE.csv"
test_side_path = "/Users/samayjain/PycharmProjects/OpenAIVers/test_side_UNIQUE.csv"
test_intensity_path = "/Users/samayjain/PycharmProjects/OpenAIVers/test_intensity_UNIQUE.csv"

df_train_top = pd.read_csv(train_top_path)
df_train_side = pd.read_csv(train_side_path)
df_train_intensity = pd.read_csv(train_intensity_path)

df_test_top = pd.read_csv(test_top_path)
df_test_side = pd.read_csv(test_side_path)
df_test_intensity = pd.read_csv(test_intensity_path)

assert len(df_train_top) == len(df_train_side) == len(df_train_intensity), "Training files have mismatched lengths!"
assert len(df_test_top) == len(df_test_side) == len(df_test_intensity), "Testing files have mismatched lengths!"

train_side = df_train_side.values[0::2, :]
train_top = df_train_top.values[0::2, :]
train_intensity = df_train_intensity.values[0::2, :]

actual_train_side = train_side[:, 1:41]
actual_train_top = train_top[:, 1:41]

independent_train = train_intensity[:, 1:]

print(f"Training Multi-Output Random Forests on {len(independent_train)} downsampled training frames...")

model_side = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_side.fit(independent_train, actual_train_side)

model_top = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_top.fit(independent_train, actual_train_top)

print("Training complete. Evaluating predictions on Train and Test sets...")

test_side = df_test_side.values
test_top = df_test_top.values
test_intensity = df_test_intensity.values

independent_test = test_intensity[:, 1:]

actual_test_side = test_side[:, 1:41]
actual_test_top = test_top[:, 1:41]

actual_test_x_side = actual_test_side[:, 0:20]
actual_test_y_side = actual_test_side[:, 20:40]
actual_test_x_top = actual_test_top[:, 0:20]
actual_test_y_top = actual_test_top[:, 20:40]

predicted_train_side = model_side.predict(independent_train)
predicted_train_top = model_top.predict(independent_train)

predicted_test_side = model_side.predict(independent_test)
predicted_test_top = model_top.predict(independent_test)

actual_train_all = np.hstack((actual_train_side, actual_train_top))
predicted_train_all = np.hstack((predicted_train_side, predicted_train_top))

actual_test_all = np.hstack((actual_test_side, actual_test_top))
predicted_test_all = np.hstack((predicted_test_side, predicted_test_top))

train_overall_r2 = r2_score(actual_train_all, predicted_train_all)
train_side_r2 = r2_score(actual_train_side, predicted_train_side)
train_top_r2 = r2_score(actual_train_top, predicted_train_top)

test_overall_r2 = r2_score(actual_test_all, predicted_test_all)
test_side_r2 = r2_score(actual_test_side, predicted_test_side)
test_top_r2 = r2_score(actual_test_top, predicted_test_top)

print("\n" + "=" * 55)
print("     FINAL PREDICTION METRICS (RANDOM FOREST)")
print("=" * 55)
print("TRAINING DATA PERFORMANCE:")
print(f"  Overall R^2 Score:  {train_overall_r2:.4f}")
print(f"  Side R^2 Score:     {train_side_r2:.4f}")
print(f"  Top R^2 Score:      {train_top_r2:.4f}")
print("-" * 55)
print("TESTING DATA PERFORMANCE:")
print(f"  Overall R^2 Score:  {test_overall_r2:.4f}")
print(f"  Side R^2 Score:     {test_side_r2:.4f}")
print(f"  Top R^2 Score:      {test_top_r2:.4f}")
print("=" * 55)
print(f"--- Processing system runtime: {time.time() - start_time:.2f} seconds total ---")

print("\nGenerating visualizations for 3 random test frames...")

num_test_samples = len(actual_test_top)
random_indices = random.sample(range(num_test_samples), 3)

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Random Forest: Actual vs Predicted Fiber Shape", fontsize=16, y=0.98)

for i, idx in enumerate(random_indices):
    actual_sx = actual_test_x_side[idx]
    actual_sy = actual_test_y_side[idx]
    actual_tx = actual_test_x_top[idx]
    actual_ty = actual_test_y_top[idx]

    pred_sx = predicted_test_side[idx, 0:20]
    pred_sy = predicted_test_side[idx, 20:40]
    pred_tx = predicted_test_top[idx, 0:20]
    pred_ty = predicted_test_top[idx, 20:40]

    axes[i, 0].plot(actual_sx, actual_sy, 'ko-', label="Actual (Ground Truth)", markersize=6, alpha=0.8)
    axes[i, 0].plot(pred_sx, pred_sy, 'ro--', label="RF Predicted", markersize=6)
    axes[i, 0].set_title(f"Test Frame {idx} - Side View")
    axes[i, 0].set_xlabel("X Coordinate")
    axes[i, 0].set_ylabel("Y Coordinate")
    axes[i, 0].legend()
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)

    axes[i, 1].plot(actual_tx, actual_ty, 'ko-', label="Actual (Ground Truth)", markersize=6, alpha=0.8)
    axes[i, 1].plot(pred_tx, pred_ty, 'ro--', label="RF Predicted", markersize=6)
    axes[i, 1].set_title(f"Test Frame {idx} - Top View")
    axes[i, 1].set_xlabel("X Coordinate")
    axes[i, 1].set_ylabel("Y Coordinate")
    axes[i, 1].legend()
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()