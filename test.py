import pandas as pd
import os

# Load the original CSV file
input_path = "/mnt/data/mumbaires.csv"   # Update if needed
df = pd.read_csv(input_path)

# Columns to remove
columns_to_remove = [
    "restaurant name",
    "rating",
    "address",
    "reviewtext",
    "reviewer rating"
]

# Drop the specified columns (ignore errors if column not found)
df_new = df.drop(columns=columns_to_remove, errors='ignore')

# Create datasets folder if it doesn't exist
output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)

# Save the new CSV file
output_path = os.path.join(output_folder, "one.csv")
df_new.to_csv(output_path, index=False)

print(f"New CSV saved successfully at: {output_path}")
