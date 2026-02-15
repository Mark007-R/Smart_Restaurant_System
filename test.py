import pandas as pd
import os

input_path = "datasets/mumbaires.csv"
df = pd.read_csv(input_path)

columns_to_keep = [
    "Restaurant Name",
    "Rating",
    "Address",
    "Review Text",
    "Reviewer Rating"
]

df_new = df[columns_to_keep]

output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "one.csv")
df_new.to_csv(output_path, index=False)

print(f"'one.csv' created successfully at: {output_path}")
