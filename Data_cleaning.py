import pandas as pd

def load_and_clean_data(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Keep only necessary columns
    df = df[['Area', 'Item', 'Element', 'Year', 'Unit', 'Value']]

    # Pivot the 'Element' column to become separate columns
    df_pivot = df.pivot_table(
        index=['Area', 'Item', 'Year'],
        columns='Element',
        values='Value',
        aggfunc='sum'
    ).reset_index()

    # Remove the column name index
    df_pivot.columns.name = None

    # Rename the crop-related columns
    df_pivot = df_pivot.rename(columns={
        'Area harvested': 'Area_Harvested',
        'Yield': 'Yield_kg_per_ha',
        'Production': 'Production_tonnes'
    })

    # Drop rows where all crop values are missing
    df_cleaned = df_pivot.dropna(subset=['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes'], how='all')

    # Fill missing values with the median
    df_cleaned = df_cleaned.fillna({
        'Area_Harvested': df_cleaned['Area_Harvested'].median(),
        'Yield_kg_per_ha': df_cleaned['Yield_kg_per_ha'].median(),
        'Production_tonnes': df_cleaned['Production_tonnes'].median()
    })

    # Keep only crop-specific columns
    df_cleaned = df_cleaned[['Area', 'Item', 'Year', 'Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']]

    # Save to a new CSV file
    df_cleaned.to_csv(output_path, index=False)
    print("Data cleaned and saved to", output_path)
    print(df_cleaned.columns)
# Example usage
if __name__ == "__main__":
    input_file = "D:/Guvi/Crop_production_prediction/Input_path.csv"
    output_file = "D:/Guvi/Crop_production_prediction/cleaned_crops_only.csv"
    load_and_clean_data(input_file, output_file)
