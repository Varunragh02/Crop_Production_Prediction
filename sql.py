import pandas as pd
from sqlalchemy import create_engine

# 🛠️ Update these values with your own
DB_USER = "root"
DB_PASSWORD = "Selvamk1403#"
DB_HOST = "localhost"
DB_NAME = "crop_production_prediction"
CSV_PATH = r"D:/Guvi/Crop_production_prediction/cleaned_crops_only.csv"

# Step 1: Load the cleaned CSV
df = pd.read_csv(CSV_PATH)
# Step 2: Display the DataFrame
print("Data loaded successfully!")
print("DataFrame columns:")
print(df.columns)

# Step 3: Connect to MySQL
engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
df.to_sql("crop_production_data", con=engine, if_exists="replace", index=False)
print("✅ Data successfully inserted into MySQL database!")

# Step 4: Insert into MySQL table
with engine.begin() as conn:
    df.to_sql('crop_production_data', con=conn, if_exists='append', index=False)
    print("✅ Data successfully inserted into 'crop_production' table!")
