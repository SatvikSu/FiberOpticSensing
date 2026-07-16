import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- SETUP & REPRODUCIBILITY ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check for GPU acceleration (CUDA for NVIDIA, MPS for Apple Silicon, CPU as fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. PYTORCH DATASET CLASS ---
class FiberDataset(Dataset):
    def __init__(self, intensity_path, side_path, top_path):
        # Load CSVs
        df_intensity = pd.read_csv(intensity_path)
        df_side = pd.read_csv(side_path)
        df_top = pd.read_csv(top_path)

        # Ensure alignments
        assert len(df_intensity) == len(df_side) == len(df_top), "Length mismatch between CSV files!"

        # Extract features (drop Frame_ID at column index 0)
        self.X = df_intensity.iloc[:, 1:].values.astype(np.float32)

        # Extract targets (drop Frame_ID, columns 1:41 contain 40 coordinates)
        y_side = df_side.iloc[:, 1:41].values.astype(np.float32)
        y_top = df_top.iloc[:, 1:41].values.astype(np.float32)

        # Concatenate targets: 40 (side) + 40 (top) = 80 targets total
        self.y = np.hstack((y_side, y_top))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# --- 2. TRANSFORMER REGRESSOR MODEL ---
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerRegressor, self).__init__()

        # Project raw input features to the transformer's inner dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Map back from d_model representation to coordinate output targets (80 outputs)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, x):
        # Transform to sequence: [batch_size, seq_len=1, d_model]
        x = self.input_projection(x).unsqueeze(1)

        # Process through transformer
        x = self.transformer_encoder(x)

        # Squeeze out the sequence dimension to output: [batch_size, d_model]
        x = x.squeeze(1)

        # Map to final targets
        out = self.output_projection(x)
        return out


# --- 3. LOAD DATA AND INITIALIZE ---
train_dataset = FiberDataset("train_intensity_9ptaverages.csv", "train_side_UNIQUE.csv", "train_top_UNIQUE.csv")
test_dataset = FiberDataset("test_intensity_9ptaverages.csv", "test_side_UNIQUE.csv", "test_top_UNIQUE.csv")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Auto-detect dimensions
input_dim = train_dataset.X.shape[1]
output_dim = train_dataset.y.shape[1]  # 80 (40 side coords + 40 top coords)

model = TransformerRegressor(input_dim=input_dim, output_dim=output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# --- 4. TRAINING LOOP ---
epochs = 100
print(f"\nTraining Transformer on {len(train_dataset)} samples for {epochs} epochs...")
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    # Evaluate Validation/Test loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(test_loader.dataset)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # Print progress every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}")

print(f"--- Training completed in {time.time() - start_time:.2f} seconds ---")

# --- 5. EVALUATION AND METRICS ---
model.eval()
all_actuals = []
all_preds = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        predictions = model(batch_x)
        all_actuals.append(batch_y.numpy())
        all_preds.append(predictions.cpu().numpy())

actual_test_all = np.vstack(all_actuals)
predicted_test_all = np.vstack(all_preds)

actual_test_side = actual_test_all[:, 0:40]
actual_test_top = actual_test_all[:, 40:80]

predicted_test_side = predicted_test_all[:, 0:40]
predicted_test_top = predicted_test_all[:, 40:80]

# --- Calculate R^2 Metrics (Closer to 1.0 is better) ---
overall_r2 = r2_score(actual_test_all, predicted_test_all)
side_r2 = r2_score(actual_test_side, predicted_test_side)
top_r2 = r2_score(actual_test_top, predicted_test_top)

# --- Calculate Distance Error Metrics (Closer to 0.0 is better) ---
mae_overall = mean_absolute_error(actual_test_all, predicted_test_all)
rmse_overall = np.sqrt(mean_squared_error(actual_test_all, predicted_test_all))

print("\n" + "=" * 55)
print("     FINAL PREDICTION METRICS (PYTORCH TRANSFORMER)")
print("=" * 55)
print("CORRELATION (Higher is better, max 1.0):")
print(f"  Overall R^2 Score:  {overall_r2:.4f}")
print(f"  Side R^2 Score:     {side_r2:.4f}")
print(f"  Top R^2 Score:      {top_r2:.4f}")
print("-" * 55)
print("PHYSICAL ERROR (Lower is better, min 0.0):")
print(f"  Mean Absolute Error (MAE):   {mae_overall:.5f} units")
print(f"  Root Mean Sq. Error (RMSE):  {rmse_overall:.5f} units")
print("=" * 55)

# --- 6. PLOT COMPARISONS (Visualizing 3 random frames) ---
print("\nGenerating visualizations...")
num_test_samples = len(actual_test_all)
random_indices = random.sample(range(num_test_samples), 3)

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Transformer: Actual vs Predicted Fiber Shape", fontsize=16, y=0.98)

for i, idx in enumerate(random_indices):
    # Slice actual and predicted X's and Y's
    actual_sx = actual_test_side[idx, 0:20]
    actual_sy = actual_test_side[idx, 20:40]
    actual_tx = actual_test_top[idx, 0:20]
    actual_ty = actual_test_top[idx, 20:40]

    pred_sx = predicted_test_side[idx, 0:20]
    pred_sy = predicted_test_side[idx, 20:40]
    pred_tx = predicted_test_top[idx, 0:20]
    pred_ty = predicted_test_top[idx, 20:40]

    # Side View
    axes[i, 0].plot(actual_sx, actual_sy, 'ko-', label="Actual (Ground Truth)", alpha=0.8)
    axes[i, 0].plot(pred_sx, pred_sy, 'bo--', label="Transformer Predicted")
    axes[i, 0].set_title(f"Test Frame {idx} - Side View")
    axes[i, 0].set_xlabel("X Coordinate")
    axes[i, 0].set_ylabel("Y Coordinate")
    axes[i, 0].legend()
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)

    # Top View
    axes[i, 1].plot(actual_tx, actual_ty, 'ko-', label="Actual (Ground Truth)", alpha=0.8)
    axes[i, 1].plot(pred_tx, pred_ty, 'bo--', label="Transformer Predicted")
    axes[i, 1].set_title(f"Test Frame {idx} - Top View")
    axes[i, 1].set_xlabel("X Coordinate")
    axes[i, 1].set_ylabel("Y Coordinate")
    axes[i, 1].legend()
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()