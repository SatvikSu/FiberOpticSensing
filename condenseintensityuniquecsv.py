import pandas as pd

train_df = pd.read_csv("train_intensity_UNIQUE.csv")
test_df = pd.read_csv("test_intensity_UNIQUE.csv")

train_filtered = train_df.loc[:, ~train_df.columns.str.contains('pt_')]
test_filtered = test_df.loc[:, ~test_df.columns.str.contains('pt_')]

train_filtered.to_csv("train_intensity_averages.csv", index=False)
test_filtered.to_csv("test_intensity_averages.csv", index=False)