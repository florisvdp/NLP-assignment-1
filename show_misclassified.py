import pandas as pd

df = pd.read_csv("results/LogisticRegression_misclassified_top20.csv")

print("Columns:", df.columns)

print("\nTop 10 full rows:")
print(df.head(10))

print("\nConcise examples for analysis:")
for i, row in df.head(10).iterrows():
    print(f"{row['true_label']} | {row['predicted_label']} | {row['text'][:100]}...")
