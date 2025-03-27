import pandas as pd
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Define the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load 2018 election data
data_2018_path = os.path.join(current_dir, '../data/election_data_2018.csv')
if not os.path.exists(data_2018_path):
    raise FileNotFoundError(f"2018 election data not found at {data_2018_path}. Please ensure the file exists.")
df_2018 = pd.read_csv(data_2018_path)

# Load 2022 election data
data_2022_path = os.path.join(current_dir, '../data/election_data_2022.csv')
if not os.path.exists(data_2022_path):
    raise FileNotFoundError(f"2022 election data not found at {data_2022_path}. Please ensure the file exists.")
df_2022 = pd.read_csv(data_2022_path)

# Load polling data
polling_data_path = os.path.join(current_dir, '../data/polling_data.csv')
if not os.path.exists(polling_data_path):
    raise FileNotFoundError(f"Polling data not found at {polling_data_path}. Please ensure the file exists.")
df_polling = pd.read_csv(polling_data_path)

# Sort polling data by 'Last Date of Polling' in descending order
df_polling = df_polling.sort_values('Last Date of Polling', ascending=False).reset_index(drop=True)

# Calculate weights (more recent polls get higher weight)
weights = np.linspace(0.5, 1.0, len(df_polling))
weights = weights / weights.sum()

# Apply weights to get polling average
polling_average = {}
for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
    if party in df_polling.columns:
        polling_average[party] = sum(df_polling[party] * weights)
    else:
        raise KeyError(f"Column '{party}' not found in polling data.")

print("\nPolling Average:")
print(polling_average)

# Calculate vote shares for 2018 and 2022
def calculate_vote_shares(df):
    # Replace zeros in Total to avoid division by zero
    df["Total"] = df["Total"].replace(0, 1)
    
    # Calculate shares - ensure Other is included
    for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
        df[f"{party}_Share"] = df[party] / df["Total"]
    
    return df

df_2018 = calculate_vote_shares(df_2018)
df_2022 = calculate_vote_shares(df_2022)

# Calculate vote shares for 2018 and 2022
def calculate_vote_shares(df):
    df["PC_Share"] = df["PC"] / df["Total"]
    df["NDP_Share"] = df["NDP"] / df["Total"]
    df["Liberal_Share"] = df["Liberal"] / df["Total"]
    df["Green_Share"] = df["Green"] / df["Total"]
    df["Other_Share"] = df["Other"] / df["Total"]
    return df

df_2018 = calculate_vote_shares(df_2018)
df_2022 = calculate_vote_shares(df_2022)

# Merge 2018 and 2022 data to calculate historical swings
df_swings = pd.merge(df_2018, df_2022, on="Riding", suffixes=("_2018", "_2022"))

# Calculate historical swings
df_swings["Swing_PC"] = df_swings["PC_Share_2022"] - df_swings["PC_Share_2018"]
df_swings["Swing_NDP"] = df_swings["NDP_Share_2022"] - df_swings["NDP_Share_2018"]
df_swings["Swing_Liberal"] = df_swings["Liberal_Share_2022"] - df_swings["Liberal_Share_2018"]
df_swings["Swing_Green"] = df_swings["Green_Share_2022"] - df_swings["Green_Share_2018"]
df_swings["Swing_Other"] = df_swings["Other_Share_2022"] - df_swings["Other_Share_2018"]

# Visualize historical swing distribution
plt.figure(figsize=(12, 8))
for i, party in enumerate(["PC", "NDP", "Liberal", "Green", "Other"]):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_swings[f"Swing_{party}"], kde=True)
    plt.title(f"{party} Vote Share Swing 2018-2022")
    plt.xlabel("Swing (percentage points)")
plt.tight_layout()
plt.savefig(os.path.join(current_dir, '../data/historical_swings.png'))

# Prepare features (X) and targets (y) for swing prediction
features = ["PC_Share_2018", "NDP_Share_2018", "Liberal_Share_2018", "Green_Share_2018", "Other_Share_2018"]
targets = ["Swing_PC", "Swing_NDP", "Swing_Liberal", "Swing_Green", "Swing_Other"]

# Train a model for each party's swing
models = {}
feature_importance = {}
for target in targets:
    print(f"Training model for {target}...")
    X = df_swings[features]
    y = df_swings[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {target}: {mse}")
    
    # Save the model
    models[target] = model
    
    # Save feature importance
    feature_importance[target] = dict(zip(features, model.feature_importances_))

# Visualize feature importance
plt.figure(figsize=(12, 8))
for i, target in enumerate(targets):
    plt.subplot(2, 3, i+1)
    importance = feature_importance[target]
    bars = plt.bar(range(len(importance)), list(importance.values()))
    plt.xticks(range(len(importance)), list(importance.keys()), rotation=45)
    plt.title(f"Feature Importance for {target}")
plt.tight_layout()
plt.savefig(os.path.join(current_dir, '../data/feature_importance.png'))

# Get polling data - use weighted average of recent polls instead of just the latest
df_polling = df_polling.sort_values('Last Date of Polling', ascending=False).reset_index(drop=True)
# Calculate weights (more recent polls get higher weight)
weights = np.linspace(0.5, 1.0, len(df_polling))
weights = weights / weights.sum()

# Apply weights to get polling average
polling_average = {}
for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
    polling_average[party] = sum(df_polling[party] * weights)

print("\nPolling Average:")
print(polling_average)

# Calculate province-wide vote shares from 2022 data
province_2022 = {
    "PC_Share": df_2022["PC"].sum() / df_2022["Total"].sum(),
    "NDP_Share": df_2022["NDP"].sum() / df_2022["Total"].sum(),
    "Liberal_Share": df_2022["Liberal"].sum() / df_2022["Total"].sum(),
    "Green_Share": df_2022["Green"].sum() / df_2022["Total"].sum(),
    "Other_Share": df_2022["Other"].sum() / df_2022["Total"].sum()
}

# Calculate swings from polling data (convert percentages to proportions)
polling_swings = {
    "Swing_PC": polling_average["PC"] / 100 - province_2022["PC_Share"],
    "Swing_NDP": polling_average["NDP"] / 100 - province_2022["NDP_Share"],
    "Swing_Liberal": polling_average["Liberal"] / 100 - province_2022["Liberal_Share"],
    "Swing_Green": polling_average["Green"] / 100 - province_2022["Green_Share"],
    "Swing_Other": polling_average["Other"] / 100 - province_2022["Other_Share"]
}

print("\nCalculated Polling Swings:")
for party, swing in polling_swings.items():
    print(f"{party}: {swing:.4f}")

# Use the trained models to predict riding-specific swings
df_2022_features = df_2022[["PC_Share", "NDP_Share", "Liberal_Share", "Green_Share", "Other_Share"]]
df_2022_features.columns = features  # Rename columns to match model features

for target, model in models.items():
    # Predict riding-specific swings
    df_2022[target] = model.predict(df_2022_features)
    
    # Visualize the distribution of predicted swings
    plt.figure(figsize=(10, 6))
    sns.histplot(df_2022[target], kde=True)
    plt.axvline(polling_swings[target], color='r', linestyle='--', 
               label=f'Polling Swing: {polling_swings[target]:.4f}')
    plt.title(f"Distribution of Predicted {target} by Riding")
    plt.legend()
    plt.savefig(os.path.join(current_dir, f'../data/predicted_{target}_distribution.png'))
    
    # Adjust the predictions to match the polling swings
    # Calculate the average predicted swing
    avg_predicted_swing = df_2022[target].mean()
    
    # Calculate adjustment needed to match polling
    adjustment = polling_swings[target] - avg_predicted_swing
    
    # Apply the adjustment
    df_2022[target] += adjustment

# Apply predicted swings to get 2025 vote shares
for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
    df_2022[f"Predicted_{party}_Share"] = df_2022[f"{party}_Share"] + df_2022[f"Swing_{party}"]

# Ensure vote shares are between 0 and 1
for col in ["Predicted_PC_Share", "Predicted_NDP_Share", "Predicted_Liberal_Share", "Predicted_Green_Share", "Predicted_Other_Share"]:
    df_2022[col] = df_2022[col].clip(0, 1)

# In your projection calculation code:
vote_cols = ["Predicted_PC_Share", "Predicted_NDP_Share", 
             "Predicted_Liberal_Share", "Predicted_Green_Share", 
             "Predicted_Other_Share"]

# Before normalization, enforce minimum values
MIN_OTHER_SHARE = 0.01  # 1% minimum
df_2022["Predicted_Other_Share"] = df_2022["Predicted_Other_Share"].clip(lower=MIN_OTHER_SHARE)

# Then normalize
df_2022["Total_Predicted"] = df_2022[vote_cols].sum(axis=1)
for col in vote_cols:
    df_2022[col] = df_2022[col] / df_2022["Total_Predicted"]

# Monte Carlo simulation for uncertainty
n_sims = 1000
seat_counts = {party: [] for party in ["PC", "NDP", "Liberal", "Green", "Other"]}
winners_by_riding = {riding: {"PC": 0, "NDP": 0, "Liberal": 0, "Green": 0, "Other": 0} 
                    for riding in df_2022["Riding"]}

for i in range(n_sims):
    # Copy the dataframe
    sim_df = df_2022.copy()
    
    # Add random noise to vote shares
    noise_sd = 0.03
    for col in vote_cols:
        # Use smaller noise for "Other" to preserve small values
        current_sd = noise_sd * (0.3 if "Other" in col else 1.0)
        noise = np.random.normal(0, current_sd, len(sim_df))
        sim_df[col] = (sim_df[col] + noise).clip(0, 1)
    
    # Ensure vote shares are between 0 and 1
    for col in vote_cols:
        sim_df[col] = sim_df[col].clip(0, 1)
    
    # Normalize
    sim_df["Total_Predicted"] = sim_df[vote_cols].sum(axis=1)
    for col in vote_cols:
        sim_df[col] = sim_df[col] / sim_df["Total_Predicted"]
    
    # Determine winners
    for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
        party_col = f"Predicted_{party}_Share"
        
        # Get the max vote share for each riding
        sim_df["max_vote"] = sim_df[vote_cols].max(axis=1)
        
        # Party wins if they have the max vote share
        sim_df[f"{party}_wins"] = (sim_df[party_col] == sim_df["max_vote"]).astype(int)
    
    # Count seats won by each party
    for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
        seats_won = sim_df[f"{party}_wins"].sum()
        seat_counts[party].append(seats_won)
        
        # Record winners by riding
        for _, row in sim_df.iterrows():
            if row[f"{party}_wins"] == 1:
                winners_by_riding[row["Riding"]][party] += 1

# Calculate confidence intervals
confidence_intervals = {}
for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
    seats = np.array(seat_counts[party])
    mean = np.mean(seats)
    lower = np.percentile(seats, 5)
    upper = np.percentile(seats, 95)
    confidence_intervals[party] = {'mean': mean, 'lower': lower, 'upper': upper}

# Calculate probability of majority
majority_threshold = 63
probability_of_majority = {}
for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
    probability_of_majority[party] = sum(1 for seats in seat_counts[party] if seats >= majority_threshold) / n_sims

# Save confidence intervals and probability of majority
with open('confidence_intervals.json', 'w') as f:
    json.dump(confidence_intervals, f)

with open('probability_of_majority.json', 'w') as f:
    json.dump(probability_of_majority, f)

def recalculate_polling_average(df_polling, excluded_firms=None, date_range=None, min_sample_size=800):
    """
    Recalculates polling average with filters applied
    
    Args:
        df_polling: DataFrame of polling data
        excluded_firms: List of firms to exclude
        date_range: Tuple of (start_date, end_date) as datetime objects
        min_sample_size: Minimum sample size to include
        
    Returns:
        Dictionary of party averages and the filtered DataFrame
    """
    # Make a copy to avoid modifying original
    filtered = df_polling.copy()
    
    # Apply filters
    if excluded_firms:
        filtered = filtered[~filtered["Polling Firm"].isin(excluded_firms)]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["Last Date of Polling"] >= pd.to_datetime(start_date)) &
            (filtered["Last Date of Polling"] <= pd.to_datetime(end_date))
        ]
    
    if min_sample_size:
        filtered = filtered[filtered["Sample Size"] >= min_sample_size]
    
    # If no polls remain after filtering, return None
    if len(filtered) == 0:
        return None, None
    
    # Calculate weights (more recent = higher weight)
    filtered = filtered.sort_values('Last Date of Polling', ascending=False)
    weights = np.linspace(0.5, 1.0, len(filtered))
    weights = weights / weights.sum()
    
    # Calculate weighted averages
    averages = {}
    for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
        if party in filtered.columns:
            averages[party] = np.sum(filtered[party] * weights)
        else:
            averages[party] = np.nan
    
    return averages, filtered


def update_projections(polling_avg, df_2022, models, province_2022, session_state=None):
    """
    Recalculate all projections based on new polling averages
    
    Args:
        polling_avg: Dictionary of new polling averages
        df_2022: DataFrame with 2022 election data
        models: Dictionary of trained models
        province_2022: Dictionary of 2022 province-wide shares
        session_state: Optional Streamlit session state for caching
        
    Returns:
        Dictionary with:
        - seat_projections: DataFrame of seat counts
        - confidence_intervals: Dictionary of confidence intervals
        - probability_of_majority: Dictionary of probabilities
        - predicted_data: DataFrame with predictions
    """
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Create cache key
    cache_key = f"proj_cache_{hash(str(polling_avg))}"
    
    # Check cache if session_state is provided
    if session_state and cache_key in session_state:
        return session_state[cache_key]
        
    if models is None:
        raise ValueError("No models available for projections")
    
    # Calculate new swings from polling data
    polling_swings = {
        "Swing_PC": polling_avg["PC"] / 100 - province_2022["PC_Share"],
        "Swing_NDP": polling_avg["NDP"] / 100 - province_2022["NDP_Share"],
        "Swing_Liberal": polling_avg["Liberal"] / 100 - province_2022["Liberal_Share"],
        "Swing_Green": polling_avg["Green"] / 100 - province_2022["Green_Share"],
        "Swing_Other": polling_avg["Other"] / 100 - province_2022["Other_Share"]
    }
    
    # Prepare features
    features = ["PC_Share", "NDP_Share", "Liberal_Share", "Green_Share", "Other_Share"]
    df_2022_features = df_2022[features]
    df_2022_features.columns = ["PC_Share_2018", "NDP_Share_2018", "Liberal_Share_2018", "Green_Share_2018", "Other_Share_2018"]
    
    # Predict and adjust swings
    for target, model in models.items():
        df_2022[target] = model.predict(df_2022_features)
        adjustment = polling_swings[target] - df_2022[target].mean()
        df_2022[target] += adjustment
    
    # Calculate predicted shares
    vote_cols = []
    for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
        col = f"Predicted_{party}_Share"  # This creates columns like "Predicted_PC_Share"
        df_2022[col] = (df_2022[f"{party}_Share"] + df_2022[f"Swing_{party}"]).clip(0, 1)
        vote_cols.append(col)
    
    # Normalize
    df_2022["Total_Predicted"] = df_2022[vote_cols].sum(axis=1)
    for col in vote_cols:
        df_2022[col] = df_2022[col] / df_2022["Total_Predicted"]
    
    # Monte Carlo simulation
    n_sims = 1000
    seat_counts = {party: [] for party in ["PC", "NDP", "Liberal", "Green", "Other"]}
    
    for _ in range(n_sims):
        sim_df = df_2022.copy()
        for col in vote_cols:
            noise = np.random.normal(0, 0.03, len(sim_df))
            sim_df[col] = (sim_df[col] + noise).clip(0, 1)
        
        # Normalize
        sim_df["Total_Predicted"] = sim_df[vote_cols].sum(axis=1)
        for col in vote_cols:
            sim_df[col] = sim_df[col] / sim_df["Total_Predicted"]
        
        # Count seats
        winners = sim_df[vote_cols].idxmax(axis=1)
        for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
            seat_counts[party].append((winners == f"Predicted_{party}_Share").sum())
    
    # Calculate results
    results = {
        'seat_projections': pd.DataFrame([{
            party: np.mean(seat_counts[party])
            for party in ["PC", "NDP", "Liberal", "Green", "Other"]
        }]),
        'confidence_intervals': {
            party: {
                'mean': np.mean(seats),
                'lower': np.percentile(seats, 5),
                'upper': np.percentile(seats, 95)
            }
            for party, seats in seat_counts.items()
        },
        'probability_of_majority': {
            party: sum(1 for s in seats if s >= 63) / n_sims
            for party, seats in seat_counts.items()
        },
        'predicted_data': df_2022
    }
    
    # Cache results if session state available
    if session_state:
        st.session_state[cache_key] = results
        
    return {
        'seat_projections': pd.DataFrame([{
            'PC': np.mean(seat_counts['PC']),
            'NDP': np.mean(seat_counts['NDP']),
            'Liberal': np.mean(seat_counts['Liberal']),
            'Green': np.mean(seat_counts['Green']),
            'Other': np.mean(seat_counts['Other'])
        }]),
        'confidence_intervals': {
            'PC': {
                'mean': np.mean(seat_counts['PC']),
                'lower': np.percentile(seat_counts['PC'], 5),
                'upper': np.percentile(seat_counts['PC'], 95)
            },
            'NDP': {
                'mean': np.mean(seat_counts['NDP']),
                'lower': np.percentile(seat_counts['NDP'], 5),
                'upper': np.percentile(seat_counts['NDP'], 95)
            },
            'Liberal': {
                'mean': np.mean(seat_counts['Liberal']),
                'lower': np.percentile(seat_counts['Liberal'], 5),
                'upper': np.percentile(seat_counts['Liberal'], 95)
            },
            'Green': {
                'mean': np.mean(seat_counts['Green']),
                'lower': np.percentile(seat_counts['Green'], 5),
                'upper': np.percentile(seat_counts['Green'], 95)
            },
            'Other': {
                'mean': np.mean(seat_counts['Other']),
                'lower': np.percentile(seat_counts['Other'], 5),
                'upper': np.percentile(seat_counts['Other'], 95)
            }
        },
        'probability_of_majority': {
            'PC': sum(1 for s in seat_counts['PC'] if s >= 63) / n_sims,
            'NDP': sum(1 for s in seat_counts['NDP'] if s >= 63) / n_sims,
            'Liberal': sum(1 for s in seat_counts['Liberal'] if s >= 63) / n_sims,
            'Green': sum(1 for s in seat_counts['Green'] if s >= 63) / n_sims,
            'Other': sum(1 for s in seat_counts['Other'] if s >= 63) / n_sims
        },
        'predicted_data': df_2022
    }

__all__ = ['recalculate_polling_average', 'update_projections', 'load_data']