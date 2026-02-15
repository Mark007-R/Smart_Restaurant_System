import pandas as pd
import os
import ast

# Load CSV
input_path = "datasets/zomato.csv"
df = pd.read_csv(input_path)

# Clean restaurant rating (4.1/5 → 4.1)
df["rate"] = df["rate"].astype(str).str.replace("/5", "", regex=False)

processed_data = []

for index, row in df.iterrows():
    
    name = row["name"]
    address = row["address"]
    rate = row["rate"]
    
    # Skip if reviews_list is empty or NaN
    if pd.isna(row["reviews_list"]):
        continue
    
    try:
        reviews = ast.literal_eval(row["reviews_list"])
    except:
        continue
    
    for review in reviews:
        
        # Skip invalid reviews
        if not review or review[0] is None or review[1] is None:
            continue
        
        reviewer_rating = str(review[0]).replace("Rated ", "").strip()
        review_text = str(review[1]).replace("RATED\n", "").strip()
        
        processed_data.append([
            name,
            address,
            rate,
            review_text,
            reviewer_rating
        ])

# Create new dataframe
df_new = pd.DataFrame(processed_data, columns=[
    "name",
    "address",
    "rate",
    "review_text",
    "reviewer_rating"
])

# Save file
output_folder = "datasets"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "four.csv")
df_new.to_csv(output_path, index=False)

print(f"'three.csv' created successfully at: {output_path}")
