import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load model and dataset
@st.cache_resource
def load_model_and_data():
    model = joblib.load("D:/Guvi/Crop_production_prediction/all_models.pkl")
    data = pd.read_csv("D:/Guvi/Crop_production_prediction/cleaned_crops_only.csv")
    area_dummies = pd.get_dummies(data['Area'], prefix='Area', drop_first=True)
    item_dummies = pd.get_dummies(data['Item'], prefix='Item', drop_first=True)
    dummy_columns = list(area_dummies.columns) + list(item_dummies.columns)
    return model, data, dummy_columns

model, df, dummy_columns = load_model_and_data()

# App title and description
st.title("Crop Production Prediction App")
st.markdown("This app predicts crop production (in tons) based on year, area harvested, yield, crop, and region.")

# Sidebar for inputs
st.sidebar.header("Input Parameters")
selected_area = st.sidebar.selectbox("Select Region:", sorted(df['Area'].unique()))
selected_crop = st.sidebar.selectbox("Select Crop:", sorted(df['Item'].unique()))
selected_year = st.sidebar.slider("Year", int(df['Year'].min()), int(df['Year'].max()), step=1)
area_harvested = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0, value=1000.0, step=100.0)
yield_per_ha = st.sidebar.number_input("Yield (kg/ha)", min_value=0.0, value=2000.0, step=100.0)

# Prepare input data for prediction
def create_input_dataframe(area, crop, year, harvested_area, yield_value, model_columns, dummy_columns):
    input_df = pd.DataFrame({
        'Year': [year],
        'Area_Harvested': [harvested_area],
        'Yield_kg_per_ha': [yield_value]
    })

    area_dummy = f"Area_{area}"
    crop_dummy = f"Item_{crop}"
    encoded_data = {col: 0 for col in dummy_columns}
    if area_dummy in encoded_data:
        encoded_data[area_dummy] = 1
    if crop_dummy in encoded_data:
        encoded_data[crop_dummy] = 1

    encoded_df = pd.DataFrame([encoded_data])
    final_input = pd.concat([input_df, encoded_df], axis=1)
    final_input = final_input.reindex(columns=model_columns, fill_value=0)
    return final_input

# Predict button
if st.button("Predict Production"):
    model_columns = model["random_forest"].feature_names_in_
    input_data = create_input_dataframe(
        area=selected_area,
        crop=selected_crop,
        year=selected_year,
        harvested_area=area_harvested,
        yield_value=yield_per_ha,
        model_columns=model_columns,
        dummy_columns=dummy_columns
    )
    predicted_tons = model["random_forest"].predict(input_data)[0]
    st.success(f"Estimated Crop Production: {predicted_tons:,.2f} tons")
    st.info("Based on the selected inputs and the trained Random Forest model.")

# EDA section
if st.sidebar.button("Explore EDA"):
    st.subheader("Exploratory Data Analysis")

    # Top crops
    st.subheader("Top 10 Most Common Crops")
    top_crops = df['Item'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_crops.values, y=top_crops.index, palette="viridis")
    plt.title("Top 10 Crops")
    plt.xlabel("Count")
    plt.ylabel("Crop")
    st.pyplot(plt)

    # Top regions
    st.subheader("Top 10 Most Common Regions")
    top_areas = df['Area'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_areas.values, y=top_areas.index, palette="magma")
    plt.title("Top 10 Regions")
    plt.xlabel("Count")
    plt.ylabel("Region")
    st.pyplot(plt)

    # Yearly trend
    st.subheader("Average Yearly Trends")
    yearly = df.groupby("Year")[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']].mean().reset_index()
    plt.figure(figsize=(12, 6))
    for col in ['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']:
        plt.plot(yearly['Year'], yearly[col], label=col)
    plt.title("Yearly Trends")
    plt.xlabel("Year")
    plt.ylabel("Average Values")
    plt.legend()
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot(plt)

    # Boxplots
    st.subheader("Outlier Analysis")
    plt.figure(figsize=(14, 4))
    for i, col in enumerate(['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data=df[col], color='skyblue')
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    st.pyplot(plt)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Production")
    try:
        test_data = pd.read_csv("D:/Guvi/Crop_production_prediction/cleaned_crops_only.csv")
        test_data.drop(columns=['Stocks'], inplace=True, errors='ignore')
        imputer = SimpleImputer(strategy='mean')
        test_data[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']] = imputer.fit_transform(
            test_data[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tonnes']]
        )
        test_encoded = pd.get_dummies(test_data[['Area', 'Item']], drop_first=True, sparse=True)
        test_final = pd.concat([
            test_data[['Year', 'Area_Harvested', 'Yield_kg_per_ha']],
            test_encoded,
            test_data['Production_tonnes']
        ], axis=1)
        X_all = test_final.drop('Production_tonnes', axis=1)
        y_all = test_final['Production_tonnes']
        X_all = X_all.reindex(columns=model["random_forest"].feature_names_in_, fill_value=0)
        y_pred_all = model["random_forest"].predict(X_all)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_all, y=y_pred_all, alpha=0.6, color='green')
        plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--')
        plt.xlabel("Actual Production")
        plt.ylabel("Predicted Production")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not generate plot: {e}")

# Footer
st.markdown("---")
st.caption("App created using Streamlit and Scikit-learn")
