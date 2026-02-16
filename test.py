import pandas as pd
import mysql.connector

# Load CSV
df = pd.read_csv("datasets/six.csv", encoding="latin1")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="restaurant_db"
)

cursor = conn.cursor()

# Create table manually (example)
cursor.execute("""
CREATE TABLE IF NOT EXISTS restaurants (
    name TEXT,
    address TEXT,
    rate FLOAT,
    review_text TEXT,
    reviewer_rating FLOAT
)
""")

# Insert data
for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO restaurants (name, address, rate, review_text, reviewer_rating)
    VALUES (%s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
conn.close()

print("CSV stored successfully in MySQL!")
