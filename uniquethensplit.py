import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

df_top1 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-30-30 Topview_data_normalized_features.csv")
df_side1 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-30-30 Side_data_normalized_features.csv")
df_intensity1 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-30-30 Fib_intensities_normalized.csv")

df_top2 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-42-51 Topview_data_normalized_features.csv")
df_side2 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-42-51 Side_data_normalized_features.csv")
df_intensity2 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-42-51 Fib_intensities_normalized.csv")

df_top3 = pd.read_csv("2026-05-17 22-57-14 Topview_data_normalized_features.csv")
df_side3 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-57-14 Side_data_normalized_features.csv")
df_intensity3 = pd.read_csv("/Users/samayjain/PycharmProjects/OpenAIVers/2026-05-17 22-57-14 Fib_intensities_normalized.csv")

df_top_all = pd.concat([df_top1, df_top2, df_top3], ignore_index=True)
df_side_all = pd.concat([df_side1, df_side2, df_side3], ignore_index=True)
df_intensity_all = pd.concat([df_intensity1, df_intensity2, df_intensity3], ignore_index=True)

def get_synchronized_diverse_datasets(df_top, df_side, df_intensity, threshold=0.009):
    top_coords = df_top.values[:, 1:41].astype(np.float32)
    side_coords = df_side.values[:, 1:41].astype(np.float32)
    combined_shapes = np.hstack((top_coords, side_coords))
    total_frames = len(combined_shapes)

    kept_shapes_mat = np.zeros((total_frames, 80), dtype=np.float32)
    kept_shapes_mat[0] = combined_shapes[0]
    kept_indices = [0]
    kept_count = 1

    for i in range(1, total_frames):
        current_shape = combined_shapes[i]
        active_unique_matrix = kept_shapes_mat[:kept_count]
        mae_distances = np.mean(np.abs(active_unique_matrix - current_shape), axis=1)
        if np.min(mae_distances) > threshold:
            kept_indices.append(i)
            kept_shapes_mat[kept_count] = current_shape
            kept_count += 1

    df_top_unique = df_top.iloc[kept_indices].reset_index(drop=True)
    df_side_unique = df_side.iloc[kept_indices].reset_index(drop=True)
    df_intensity_unique = df_intensity.iloc[kept_indices].reset_index(drop=True)
    return df_top_unique, df_side_unique, df_intensity_unique

SIMILARITY_THRESHOLD = 0.009
df_top_u, df_side_u, df_intensity_u = get_synchronized_diverse_datasets(
    df_top_all, df_side_all, df_intensity_all, threshold=SIMILARITY_THRESHOLD
)

original_shapes = np.hstack((df_top_all.values[:, 1:41].astype(np.float32),
                             df_side_all.values[:, 1:41].astype(np.float32)))
unique_shapes = np.hstack((df_top_u.values[:, 1:41].astype(np.float32),
                           df_side_u.values[:, 1:41].astype(np.float32)))

pca = PCA(n_components=2)
original_2d = pca.fit_transform(original_shapes)
unique_2d = pca.transform(unique_shapes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
ax1.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.1, s=5, c='blue')
ax1.set_title(f"Original Dataset ({len(original_shapes)} frames)\nDense, overlapping clusters")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")

ax2.scatter(unique_2d[:, 0], unique_2d[:, 1], alpha=0.5, s=5, c='red')
ax2.set_title(f"Unique Dataset ({len(unique_shapes)} frames)\nDuplicates pruned, even spacing")
ax2.set_xlabel("Principal Component 1")

plt.suptitle("PCA Projection of 80-Dimensional Fiber Geometries", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

total_rows = len(df_top_u)
train_stop = math.floor(total_rows / 10) * 8
buffer = math.floor(total_rows / 10 * 8.25)

print(train_stop, buffer)

train_top_u = df_top_u.head(train_stop)
test_top_u = df_top_u.iloc[buffer:]

train_side_u = df_side_u.head(train_stop)
test_side_u = df_side_u.iloc[buffer:]

train_intensity_u = df_intensity_u.head(train_stop)
test_intensity_u = df_intensity_u.iloc[buffer:]

print("TRAIN -> Top:", len(train_top_u), "| Side:", len(train_side_u), "| Intensity:", len(train_intensity_u))
print("TEST  -> Top:", len(test_top_u), "| Side:", len(test_side_u), "| Intensity:", len(test_intensity_u))

assert len(train_top_u) == len(train_side_u) == len(train_intensity_u)
assert len(test_top_u) == len(test_side_u) == len(test_intensity_u)

train_top_u.to_csv("train_top_UNIQUE.csv", index=False)
train_side_u.to_csv("train_side_UNIQUE.csv", index=False)
train_intensity_u.to_csv("train_intensity_UNIQUE.csv", index=False)

test_top_u.to_csv("test_top_UNIQUE.csv", index=False)
test_side_u.to_csv("test_side_UNIQUE.csv", index=False)
test_intensity_u.to_csv("test_intensity_UNIQUE.csv", index=False)