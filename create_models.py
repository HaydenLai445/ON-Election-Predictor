import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 

# Load your election data (update paths if needed)
df_2018 = pd.read_csv("data/election_data_2018.csv")
df_2022 = pd.read_csv("data/election_data_2022.csv")

# Calculate historical vote shares
for df in [df_2018, df_2022]:
    df["PC_Share"] = df["PC"] / df["Total"]
    df["NDP_Share"] = df["NDP"] / df["Total"]
    df["Liberal_Share"] = df["Liberal"] / df["Total"]
    df["Green_Share"] = df["Green"] / df["Total"]
    df["Other_Share"] = df["Other"] / df["Total"]

# Merge and calculate actual swings
df_swings = pd.merge(df_2018, df_2022, on="Riding", suffixes=("_2018", "_2022"))
df_swings["Swing_PC"] = df_swings["PC_Share_2022"] - df_swings["PC_Share_2018"]
df_swings["Swing_NDP"] = df_swings["NDP_Share_2022"] - df_swings["NDP_Share_2018"]
df_swings["Swing_Liberal"] = df_swings["Liberal_Share_2022"] - df_swings["Liberal_Share_2018"]
df_swings["Swing_Green"] = df_swings["Green_Share_2022"] - df_swings["Green_Share_2018"]
df_swings["Swing_Other"] = df_swings["Other_Share_2022"] - df_swings["Other_Share_2018"]

# Define features (all historical shares)
features = [
    "PC_Share_2018", 
    "NDP_Share_2018", 
    "Liberal_Share_2018", 
    "Green_Share_2018", 
    "Other_Share_2018"
]

# Create and FIT Random Forest models (key change!)
models = {
    "Swing_PC": RandomForestRegressor(n_estimators=100, random_state=42).fit(df_swings[features], df_swings["Swing_PC"]),
    "Swing_NDP": RandomForestRegressor(n_estimators=100, random_state=42).fit(df_swings[features], df_swings["Swing_NDP"]),
    "Swing_Liberal": RandomForestRegressor(n_estimators=100, random_state=42).fit(df_swings[features], df_swings["Swing_Liberal"]),
    "Swing_Green": RandomForestRegressor(n_estimators=100, random_state=42).fit(df_swings[features], df_swings["Swing_Green"]),
    "Swing_Other": RandomForestRegressor(n_estimators=100, random_state=42).fit(df_swings[features], df_swings["Swing_Other"])
}

# Save models
joblib.dump(models, "models/trained_models.joblib")
print("✅ Created and FITTED Random Forest models") 