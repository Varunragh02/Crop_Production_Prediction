import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("D:/Guvi/crop_production_prediction/cleaned_crops_only.csv")

# Drop irrelevant columns
df = df.drop(columns=["laying", "milk Animals", "Producing Animals/Slaughtered", "Stocks", "Yield_kg_per_ha/Carcass Weight"], errors='ignore')

# Convert to numeric for correlation
df["Area_Harvested"] = pd.to_numeric(df["Area_Harvested"], errors="coerce")
df["Yield_kg_per_ha"] = pd.to_numeric(df["Yield_kg_per_ha"], errors="coerce")
df["Production_tonnes"] = pd.to_numeric(df["Production_tonnes"], errors="coerce")

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# 1. Crop Distribution
top_crops = df.groupby("Item")["Area_Harvested"].sum().sort_values(ascending=False).head(10)
bottom_crops = df.groupby("Item")["Area_Harvested"].sum().sort_values(ascending=True).head(10)

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.barplot(x=top_crops.values, y=top_crops.index, palette="crest")
plt.title("Top 10 Most Cultivated Crops")

plt.subplot(2, 1, 2)
sns.barplot(x=bottom_crops.values, y=bottom_crops.index, palette="flare")
plt.title("Bottom 10 Least Cultivated Crops")
plt.tight_layout()
plt.show()

# 2. Temporal Trends
yearly_stats = df.groupby("Year")[["Area_Harvested", "Yield_kg_per_ha", "Production_tonnes"]].agg({
    "Area_Harvested": "sum",
    "Yield_kg_per_ha": "mean",
    "Production_tonnes": "sum"
}).reset_index()

sns.lineplot(data=yearly_stats, x="Year", y="Area_Harvested", label="Area_Harvested", marker="o")
sns.lineplot(data=yearly_stats, x="Year", y="Production_tonnes", label="Production_tonnes", marker="o")
plt.title("Yearly Trends in Area and Production")
plt.show()

sns.lineplot(data=yearly_stats, x="Year", y="Yield_kg_per_ha", label="Average Yield", marker="o", color="orange")
plt.title("Yearly Trend in Average Yield")
plt.show()

# 3. Environmental Relationship
sns.scatterplot(data=df, x="Area_Harvested", y="Yield_kg_per_ha", hue="Item", alpha=0.5, legend=False)
plt.title("Area Harvested vs Yield")
plt.show()

# 4. Input-Output Relationship
sns.scatterplot(data=df, x="Area_Harvested", y="Production_tonnes", hue="Item", alpha=0.5, legend=False)
plt.title("Area Harvested vs Production")
plt.show()

# 5. Correlation Matrix
df_corr = df[["Area_Harvested", "Yield_kg_per_ha", "Production_tonnes"]].dropna()
print("Correlation Table:\n", df_corr.corr())

sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 6. Comparative Analysis
avg_yield = df.groupby("Item")["Yield_kg_per_ha"].mean().sort_values(ascending=False).head(10)
high_prod = df.groupby("Area")["Production_tonnes"].sum().sort_values(ascending=False).head(10)

sns.barplot(x=avg_yield.values, y=avg_yield.index)
plt.title("Top 10 High-Yield Crops")
plt.show()

sns.barplot(x=high_prod.values, y=high_prod.index)
plt.title("Top 10 Productive Regions")
plt.show()

# 7. Productivity Analysis
df["Productivity Ratio"] = df["Production_tonnes"] / df["Area_Harvested"]

# Filter out rows with NaN or infinite productivity ratios
df = df[df["Productivity Ratio"].notna() & (df["Productivity Ratio"] != float("inf"))]

# Calculate the top 10 crops by productivity
top_prod_ratio = df.groupby("Item")["Productivity Ratio"].mean().sort_values(ascending=False).head(10)

# Debug: print to verify data
print("Top 10 Crops by Productivity:\n", top_prod_ratio)

# Plot
sns.barplot(x=top_prod_ratio.values, y=top_prod_ratio.index)
plt.title("Top 10 Crops by Productivity (Production / Area)")
plt.tight_layout()  # Makes sure labels are properly placed
plt.show()

# 8. Outlier Detection in Yield
plt.figure(figsize=(16, 6))
sns.boxplot(data=df, x="Item", y="Yield_kg_per_ha")
plt.xticks(rotation=90)
plt.title("Outliers in Yield across Crops")
plt.show()
