import pandas as pd
import os

# Load the original CSV file
input_path = "/mnt/data/mumbaires.csv"   # Change path if needed
df = pd.read_csv(input_path)

# Columns to keep
columns_to_keep = [
    "Restaurant Name",
    "Rating",
    "Address",
    "Review Text",
    "Reviewer Rating"
]

# Select only the required columns
df_new = df[columns_to_keep]

# Create datasets folder if it doesn't exist
output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)

# Save the new CSV file
output_path = os.path.join(output_folder, "one.csv")
df_new.to_csv(output_path, index=False)

print(f"'one.csv' created successfully at: {output_path}")
