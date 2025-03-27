# Core imports
import streamlit as st
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import base64
import bcrypt
from datetime import datetime
from PIL import Image
from streamlit.components.v1 import html
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from streamlit import config

config.set_option("theme.base", "light")
config.set_option("theme.primaryColor", "#0068c9")
config.set_option("theme.secondaryBackgroundColor", "#f0f2f6")
config.set_option("theme.textColor", "#262730")

# Page configuration 
st.set_page_config(
    page_title="Ontario Election Predictor",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from scripts.load_data import (
    recalculate_polling_average, 
    update_projections,
)

# PDF conversion
def convert_df_to_pdf(df):
    """Convert dataframe to PDF using reportlab"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Convert dataframe to list of lists for ReportLab
    data = [df.columns.to_list()] + df.values.tolist()
    
    # Create table
    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ])
    table.setStyle(style)
    
    # Build PDF
    pdf.build([table])
    buffer.seek(0)
    return buffer

# Define party colors 
party_colors = {
    "PC": "#0070C0",  # Blue
    "NDP": "#FF8C00",  # Orange
    "Liberal": "#FF0000",  # Red
    "Green": "#00B050",  # Green
    "Other": "#808080"   # Gray
}


# Database functions
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB,
            last_login TEXT,
            login_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def register_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, last_login, login_count) VALUES (?, ?, datetime('now'), 1)", 
                 (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode(), result[0]):
        update_login_stats(username)
        return True
    return False

def update_login_stats(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = datetime('now'), login_count = login_count + 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def get_user_stats(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT last_login, login_count FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return {
        "last_login": result[0],
        "login_count": result[1]
    } if result else None

# Initialize app
init_db()

# Session state setup
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_riding' not in st.session_state:  
    st.session_state.selected_riding = None  
if 'trusted_firms' not in st.session_state:
    st.session_state.trusted_firms = []
if 'poll_recency_weight' not in st.session_state:
    st.session_state.poll_recency_weight = 0.7
if 'uncertainty_factor' not in st.session_state:
    st.session_state.uncertainty_factor = 1.0
if 'show_confidence_intervals' not in st.session_state:
    st.session_state.show_confidence_intervals = True
if 'projections' not in st.session_state:
    st.session_state.projections = None

# Main app UI
st.title("Ontario Election Predictor")
st.markdown("""
This application uses machine learning to predict outcomes for the upcoming Ontario provincial election.
""")

# Sidebar - Navigation and Authentication
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page:",
        ["Dashboard", "Riding Map", "Riding Analysis", "Methodology"],
        key="main_nav"
    )
    
    st.header("Authentication")
    
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", key="login_btn"):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="conf_pass")
            
            if st.button("Register", key="reg_btn"):
                if new_pass != confirm_pass:
                    st.error("Passwords don't match")
                elif len(new_pass) < 6:
                    st.error("Password too short")
                elif register_user(new_user, new_pass):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username exists")
    else:
        user_stats = get_user_stats(st.session_state.username)
        st.success(f"Welcome {st.session_state.username}!")
        if user_stats:
            st.caption(f"Last login: {user_stats['last_login']}")
            st.caption(f"Logins: {user_stats['login_count']}")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

def calculate_seat_projections(predicted_data):
    # Include Other in the calculation
    seat_columns = ['PC_Share', 'NDP_Share', 'Liberal_Share', 'Green_Share', 'Other_Share']
    
    # Calculate mean shares with protection for small values
    mean_shares = {
        "PC": predicted_data["PC_Share"].mean(),
        "NDP": predicted_data["NDP_Share"].mean(),
        "Liberal": predicted_data["Liberal_Share"].mean(),
        "Green": predicted_data["Green_Share"].mean(),
        "Other": max(predicted_data["Other_Share"].mean(), 0.03)  # At least 3%
    }
    
    # Normalize to sum to 1 (in case we increased Other)
    total = sum(mean_shares.values())
    normalized_shares = {k: v/total for k, v in mean_shares.items()}
    
    # Calculate seats
    total_seats = 124
    seats = {
        party: int(round(normalized_shares[party] * total_seats))
        for party in normalized_shares
    }
    
    # Handle rounding errors
    while sum(seats.values()) != total_seats:
        diff = total_seats - sum(seats.values())
        largest_party = max(seats, key=seats.get)
        seats[largest_party] += diff
    
    return seats

# Main app content
if st.session_state.logged_in:
    @st.cache_data
    def load_data():
        """
        Load all required data files for the election predictor app.
        
        Returns:
            tuple: (polls_df, ridings_df, seat_projections_df, polling_swings, 
                confidence_intervals, probability_of_majority)
        """
        try:
            # Get the current directory and parent directory paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            data_dir = os.path.join(parent_dir, 'data')
            
            # Define all file paths
            data_files = {
                'predicted_data': os.path.join(data_dir, 'predicted_seats_2025.csv'),
                'polling_data': os.path.join(data_dir, 'polling_data.csv'),
                'polling_swings': os.path.join(data_dir, 'polling_swings.json'),
                'confidence_intervals': os.path.join(data_dir, 'confidence_intervals.json'),
                'probability_of_majority': os.path.join(data_dir, 'probability_of_majority.json')
            }
            
            # Check if all files exist
            for file_name, file_path in data_files.items():
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file not found: {file_path}")

            # Load predicted riding data
            predicted_data = pd.read_csv(data_files['predicted_data'])
            
            # Ensure we have the predicted winner column
            if 'Predicted_Winner' not in predicted_data.columns:
                vote_share_cols = ['Predicted_PC_Share', 'Predicted_NDP_Share', 
                                'Predicted_Liberal_Share', 'Predicted_Green_Share', 
                                'Predicted_Other_Share']
                if all(col in predicted_data.columns for col in vote_share_cols):
                    predicted_data['Predicted_Winner'] = predicted_data[vote_share_cols].idxmax(axis=1)
                    predicted_data['Predicted_Winner'] = predicted_data['Predicted_Winner'].str.replace('Predicted_', '').str.replace('_Share', '')
                else:
                    raise ValueError("Missing predicted vote share columns in data")

            # Create seat projections from the predicted data
            seat_projections = calculate_seat_projections(predicted_data)
            
            # Convert to DataFrame with date
            seat_projections_df = pd.DataFrame([{
                'Date': pd.to_datetime('today').strftime('%Y-%m-%d'),
                'PC': seat_projections['PC'],
                'NDP': seat_projections['NDP'],
                'Liberal': seat_projections['Liberal'],
                'Green': seat_projections['Green'],
                'Other': seat_projections['Other']
            }])
            
            # Load polling data
            polling_data = pd.read_csv(data_files['polling_data'])
            polling_data["Last Date of Polling"] = pd.to_datetime(polling_data["Last Date of Polling"])
            
            # Load JSON files
            def load_json_file(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            
            polling_swings = load_json_file(data_files['polling_swings'])
            confidence_intervals = load_json_file(data_files['confidence_intervals'])
            probability_of_majority = load_json_file(data_files['probability_of_majority'])
            
            return (
                polling_data,
                predicted_data,
                seat_projections_df,
                polling_swings,
                confidence_intervals,
                probability_of_majority
            )
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    # Load data
    polls_df, ridings_df, seat_projections_df, polling_swings, confidence_intervals, probability_of_majority = load_data()

    # Update projections in session state
    if st.session_state.projections is None:
        st.session_state.projections = {
            'seat_projections': seat_projections_df,
            'polling_swings': polling_swings,
            'confidence_intervals': confidence_intervals,
            'probability_of_majority': probability_of_majority
        }

    # Initialize models and province data
    if 'models' not in st.session_state:
        models_path = os.path.join(root_path, 'models', 'trained_models.joblib')
        
        try:
            # Load pre-trained models
            if os.path.exists(models_path):
                st.session_state.models = joblib.load(models_path)
                print("Models loaded successfully from:", models_path)
            else:
                # Create and FIT Random Forest fallback models
                X_dummy = [[0.3] * 5 for _ in range(124)]  # 5 dummy features matching expected input shape
                y_dummy = [0.01 * (i % 3) for i in range(124)]  # Dummy swings

                st.session_state.models = {
                    "Swing_PC": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_dummy, y_dummy),
                    "Swing_NDP": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_dummy, y_dummy),
                    "Swing_Liberal": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_dummy, y_dummy),
                    "Swing_Green": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_dummy, y_dummy),
                    "Swing_Other": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_dummy, y_dummy)
                }

                # Update the warning message:
                st.warning("Trained models not found. Using fallback Random Forest models.")

        except Exception as e:
            st.error(f"Critical error loading models: {str(e)}")
            st.stop()  # Halt the app if models can't load

        # Now this can run safely
        if 'province_2022' not in st.session_state:
            st.session_state.province_2022 = {
                "PC_Share": ridings_df["PC_Share"].mean(),
                "NDP_Share": ridings_df["NDP_Share"].mean(),
                "Liberal_Share": ridings_df["Liberal_Share"].mean(),
                "Green_Share": ridings_df["Green_Share"].mean(),
                "Other_Share": ridings_df["Other_Share"].mean()
            }

    # Update trusted_firms if it's empty (first run)
    if not st.session_state.trusted_firms:
        st.session_state.trusted_firms = polls_df["Polling Firm"].unique().tolist()

    # Dashboard page
    if page == "Dashboard":
        st.header("Ontario Election Projection Dashboard")

        # Initialize session state for filters and projections
        if 'active_polls' not in st.session_state:
            st.session_state.active_polls = polls_df.copy()  # Start with all polls
            st.session_state.projections = None

        # Always recalculate projections from current active polls
        def calculate_current_projections():
            poll_avg, _ = recalculate_polling_average(st.session_state.active_polls)
            return update_projections(
                polling_avg=poll_avg,
                df_2022=ridings_df,
                models=st.session_state.models,
                province_2022=st.session_state.province_2022
            )

        # Calculate current projections
        current_projections = calculate_current_projections()
        latest_projection = current_projections['seat_projections'].iloc[-1].to_dict()
        confidence_intervals = current_projections['confidence_intervals']
        prob_majority = current_projections['probability_of_majority']

        # Poll filtering section
        st.subheader("Filter Polls")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get all polling firms
            all_firms = polls_df["Polling Firm"].unique().tolist()
            
            # Display multiselect for polling firms
            selected_firms = st.multiselect(
                "Select polling firms to include:",
                options=all_firms,
                default=all_firms,
                key="poll_filter_firms"
            )
            
            # Date range selector
            min_date = pd.to_datetime(polls_df["Last Date of Polling"].min()).date()
            max_date = pd.to_datetime(polls_df["Last Date of Polling"].max()).date()
            
            date_range = st.slider(
                "Select date range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            
            if st.button("Apply Filters"):
                # Apply filters to get the active polls
                filtered_polls = polls_df[
                    (polls_df["Polling Firm"].isin(selected_firms)) &
                    (polls_df["Last Date of Polling"] >= pd.to_datetime(date_range[0])) &
                    (polls_df["Last Date of Polling"] <= pd.to_datetime(date_range[1]))
                ]
                
                if len(filtered_polls) == 0:
                    st.warning("No polls match your filters")
                else:
                    # Update active polls and force recalculation
                    st.session_state.active_polls = filtered_polls
                    st.rerun()
            
            if st.button("Reset to All Polls"):
                st.session_state.active_polls = polls_df.copy()
                st.rerun()

        with col2:
                st.subheader("Projected Seat Count")
                
                # Get the data
                party_order = ["PC", "NDP", "Liberal", "Green", "Other"]
                data = []
                for party in party_order:
                    data.append({
                        "Party": party,
                        "Projected": confidence_intervals[party]['mean'],
                        "Low": confidence_intervals[party]['lower'],
                        "High": confidence_intervals[party]['upper'],
                        "Color": party_colors[party]
                    })
                
                # Create two columns: bars on left, numbers on right
                col_graph, col_numbers = st.columns([3, 2])
                
                with col_graph:
                    # Simple bar chart
                    fig = go.Figure()
                    for party in data:
                        fig.add_trace(go.Bar(
                            y=[party["Party"]],
                            x=[party["Projected"]],
                            marker_color=party["Color"],
                            width=0.6,
                            name=party["Party"],
                            orientation='h'
                        ))
                    
                    # Add majority line
                    fig.add_vline(
                        x=63,
                        line=dict(color="red", width=2, dash="dash"),
                        annotation=dict(
                            text="Majority (63 seats)",
                            font=dict(size=12, color="red"),
                            y=1,
                            xanchor="left",
                            showarrow=False
                        )
                    )
                    
                    fig.update_layout(
                        height=350,
                        showlegend=False,
                        xaxis=dict(title="Number of Seats", range=[0, 100]),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=20, r=20, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_numbers:
                    # Numbers display
                    st.markdown("""
                    <style>
                    .projection-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                        font-family: Arial, sans-serif;
                    }
                    .projection-table th {
                        background-color: #f0f2f6;
                        padding: 10px;
                        text-align: left;
                    }
                    .projection-table td {
                        padding: 10px;
                        border-bottom: 1px solid #eee;
                    }
                    .party-name {
                        font-weight: bold;
                    }
                    </style>
                    
                    <table class="projection-table">
                    <tr>
                        <th>Party</th>
                        <th>Projected</th>
                        <th>Range</th>
                    </tr>
                    """, unsafe_allow_html=True)
                    
                    for party in data:
                        st.markdown(f"""
                        <tr>
                            <td><span class="party-name" style="color:{party['Color']}">{party['Party']}</span></td>
                            <td>{party['Projected']:.0f}</td>
                            <td>{party['Low']:.0f}-{party['High']:.0f}</td>
                        </tr>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    </table>
                    <div style="margin-top: 15px; font-size: 13px; color: #666;">
                    Range shows 90% confidence interval
                    </div>
                    """, unsafe_allow_html=True)
            
                # Probability display
                st.subheader("Probability of Winning Majority")
                cols = st.columns(5)
                for i, party in enumerate(party_order):
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {party_colors[party]};
                                color: white;
                                padding: 10px;
                                border-radius: 8px;
                                text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <div style="font-size: 14px; font-weight: bold;">{party}</div>
                                <div style="font-size: 20px; margin: 5px 0;">{prob_majority[party]*100:.0f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <p style="font-size: 14px; color: #555; margin-bottom: 5px;">
                        Generated through 1,000 Monte Carlo simulations with controlled randomness. Therefore, small variations may occur with identical inputs due to simulation variance. However, these variations should never change the rounded outcomes. See Methodology for full technical details
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Polling Trends Over Time")
        
                # Convert polling percentages to numeric values
                polls_vis_df = polls_df.copy()
                for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
                    polls_vis_df[party] = pd.to_numeric(polls_vis_df[party], errors='coerce')
                
                # Create interactive time series plot
                fig = px.line(polls_vis_df, 
                            x="Last Date of Polling", 
                            y=["PC", "NDP", "Liberal", "Green", "Other"],
                            color_discrete_map=party_colors,
                            labels={"value": "Support (%)", "variable": "Party"},
                            hover_data={"Polling Firm": True, "Sample Size": True})
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Polling Percentage",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data Export Section 
                st.subheader("Download Data")
            
                # Create PDF export buttons in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.expander("📄 Download Polling Data"):
                        st.download_button(
                            label="All Polls (PDF)",
                            data=convert_df_to_pdf(polls_df),
                            file_name="ontario_polls_data.pdf",
                            mime="application/pdf"
                        )
                        st.download_button(
                            label="All Polls (CSV)",
                            data=polls_df.to_csv(index=False),
                            file_name="ontario_polls_data.csv",
                            mime="text/csv"
                        )
                            
                with col2:
                    with st.expander("🗳️ Download Riding Data"):
                        st.download_button(
                            label="All Ridings (CSV)",
                            data=ridings_df.to_csv(index=False),
                            file_name="ontario_riding_data.csv",
                            mime="text/csv"
                        )

                with col3:
                    with st.expander("📊 Download Projections"):
                        st.download_button(
                            label="Full Results (PDF)",
                            data=convert_df_to_pdf(current_projections['seat_projections']),
                            file_name="ontario_projection_results.pdf",
                            mime="application/pdf"
                        )
                        st.download_button(
                            label="Full Results (CSV)",
                            data=current_projections['seat_projections'].to_csv(index=False),
                            file_name="ontario_projection_results.csv",
                            mime="text/csv"
                        )

    # Riding Analysis page
    elif page == "Riding Analysis":
        st.header("Riding-Level Analysis")

        # Load the predicted data
        _, ridings_df, _, _, _, _ = load_data()

        # Use the selected riding from session state
        if st.session_state.selected_riding:
            selected_riding = st.session_state.selected_riding
        else:
            selected_riding = st.selectbox(
                "Select a riding:",
                options=sorted(ridings_df["Riding"].unique()))
        
        # Get riding data
        riding_data = ridings_df[ridings_df["Riding"] == selected_riding].iloc[0]

        # Display predicted vote shares
        for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
            pred_share_col = f"Predicted_{party}_Share"
            
            # Safely get the value with NaN handling
            vote_share = riding_data.get(pred_share_col, 0)  # Default to 0 if column missing
            if pd.isna(vote_share):
                vote_share = 0  # Explicitly handle NaN
            
            # Format the display - multiply by 100 for percentage
            vote_share_text = f"{vote_share * 100:.1f}%"
            st.write(f"{party}: {vote_share_text}")
            
        # Create two columns - one for percentages, one for the graph
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display riding information
            st.subheader("Projected Results")
            
            # Display all vote shares with proper NaN handling
            for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
                share = riding_data.get(f"Predicted_{party}_Share", 0)
                if pd.isna(share):
                    share = 0
                st.write(f"{party}: {share*100:.1f}%")

            # Star candidate controls
            st.subheader("Star Candidate Analysis")
            enable_star = st.checkbox("Enable Star Candidate", key="star_candidate_enabled")
            
            if enable_star:
                star_party = st.selectbox(
                    "Star Candidate Party:",
                    options=["PC", "NDP", "Liberal", "Green"],
                    key="star_party_select"
                )
                
                # Calculate maximum possible boost
                current_share = riding_data.get(f"Predicted_{star_party}_Share", 0)
                if pd.isna(current_share):
                    current_share = 0
                max_boost = (1.0 - current_share) * 100  # Convert to percentage points
                
                star_boost = st.slider(
                    "Star Candidate Boost (percentage points):",
                    min_value=0.1,
                    max_value=float(np.round(max_boost, 1)),  # Dynamic max based on current share
                    value=min(5.0, max_boost),  # Default to 5% or max available
                    step=0.1,
                    key="star_boost_slider"
                ) / 100  # Convert to decimal

        with col2:
            parties = ["PC", "NDP", "Liberal", "Green", "Other"]
            colors = ["blue", "orange", "red", "green", "gray"]
            
            fig = go.Figure()

            # Current results
            for i, party in enumerate(parties):
                share = riding_data.get(f"Predicted_{party}_Share", 0)
                if pd.isna(share):
                    share = 0
                fig.add_trace(go.Bar(
                    x=[party],
                    y=[share * 100],
                    name=f"Projected {party}",
                    marker_color=colors[i],
                    width=0.3,
                    offset=-0.2
                ))

            # Star candidate projections
            if enable_star and 'star_party' in locals():
                # Get current shares with NaN handling
                current_shares = {}
                for party in parties:
                    share = riding_data.get(f"Predicted_{party}_Share", 0)
                    current_shares[party] = 0 if pd.isna(share) else share

                # Apply star candidate boost
                boosted_shares = current_shares.copy()
                boosted_shares[star_party] += star_boost

                # Calculate total votes to redistribute from other parties
                total_to_remove = star_boost
                other_parties = [p for p in parties if p != star_party]
                other_shares = {p: current_shares[p] for p in other_parties}
                total_other = sum(other_shares.values())

                # Redistribute proportionally if there are other votes
                if total_other > 0:
                    for party in other_parties:
                        proportion = other_shares[party] / total_other
                        reduction = total_to_remove * proportion
                        boosted_shares[party] = max(0, boosted_shares[party] - reduction)

                # Normalize to ensure exactly 100%
                total = sum(boosted_shares.values())
                if abs(total - 1.0) > 0.0001:
                    for party in boosted_shares:
                        boosted_shares[party] /= total

                # Add boosted results to chart
                for i, party in enumerate(parties):
                    fig.add_trace(go.Bar(
                        x=[party],
                        y=[boosted_shares[party] * 100],
                        name=f"With Star Candidate",
                        marker_color=colors[i],
                        marker_pattern_shape='/',
                        width=0.3,
                        offset=0.2
                    ))

                # Debug output
                st.write(f"Total after adjustment: {sum(boosted_shares.values())*100:.2f}%")

            fig.update_layout(
                title=f"Vote Share Analysis - {selected_riding}",
                xaxis_title="Party",
                yaxis_title="Vote Share (%)",
                barmode="group",
                legend_title="Scenario",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""Note: 'Other' votes (independents/small parties) are modeled at the provincial level and with the standard riding distribution in the dashboard but often show 0% in riding projections due to their unpredictable, hyper-local nature. Please see our methodology page for more info""")

    #Riding Map page
    elif page == "Riding Map":
        st.header("Riding Map")
        
        # Use the predicted data from load_data()
        _, predicted_df, _, _, _, _ = load_data()
        
        # Ensure we have the predicted winner column
        if 'Predicted_Winner' not in predicted_df.columns:
            st.error("Predicted winner data not available in the dataset")
            st.stop()
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filter by projected winner
            parties = ["All Parties", "PC", "NDP", "Liberal", "Green", "Other"]
            selected_party = st.selectbox("Filter by projected winner:", parties)
        
        with col2:
            # Search by riding name
            search_term = st.text_input("Search riding:", "")
        
        # Apply filters to riding data
        filtered_ridings = predicted_df.copy()
        
        if selected_party != "All Parties":
            filtered_ridings = filtered_ridings[filtered_ridings["Predicted_Winner"] == selected_party]
        
        if search_term:
            filtered_ridings = filtered_ridings[filtered_ridings["Riding"].str.contains(search_term, case=False)]
        
        # Create tabs for different visualization options
        tab1, tab2, tab3 = st.tabs(["Grid View", "Table View", "Competitive Ridings"])  # Added new tab
        
        with tab1:
            # Display ridings in a grid using predicted data
            st.subheader(f"Showing {len(filtered_ridings)} ridings")
            num_cols = 4
            
            for i in range(0, len(filtered_ridings), num_cols):
                cols = st.columns(num_cols)
                
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(filtered_ridings):
                        riding = filtered_ridings.iloc[idx]
                        with cols[j]:
                            party = riding["Predicted_Winner"]
                            color = party_colors.get(party, "#808080")
                            
                            # Get predicted vote share
                            vote_share_col = f"Predicted_{party}_Share"
                            vote_share_text = ""
                            if vote_share_col in riding:
                                vote_share = riding[vote_share_col] * 100
                                vote_share_text = f"{vote_share:.1f}%"
                            
                            st.markdown(f"""
                            <div style="
                                border-left: 5px solid {color};
                                background-color: #f8f9fa;
                                padding: 10px;
                                margin-bottom: 10px;
                                border-radius: 3px;
                                min-height: 80px;
                                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                                cursor: pointer;
                                <p style="font-weight: bold; margin-bottom: 5px; font-size: 14px;">
                                    {riding['Riding']}
                                </p>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;">
                                        {party}
                                    </span>
                                    <span style="font-weight: bold; font-size: 16px;">
                                        {vote_share_text}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab2:
            # Table view using predicted data
            display_cols = ["Riding", "Predicted_Winner"]
            for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
                share_col = f"Predicted_{party}_Share"
                if share_col in predicted_df.columns:
                    display_cols.append(share_col)
            
            table_data = filtered_ridings[display_cols].copy()
           
            # Format vote shares as percentages
            for party in ["PC", "NDP", "Liberal", "Green", "Other"]:
                share_col = f"Predicted_{party}_Share"
                if share_col in table_data.columns:
                    table_data[share_col] = table_data[share_col].apply(lambda x: f"{x*100:.1f}%")
            
            # Rename columns for display
            table_data = table_data.rename(columns={
                "Predicted_Winner": "Projected Winner",
                "Predicted_PC_Share": "PC %",
                "Predicted_NDP_Share": "NDP %",
                "Predicted_Liberal_Share": "Liberal %",
                "Predicted_Green_Share": "Green %",
                "Predicted_Other_Share": "Other %"
            })
            
            st.dataframe(
                table_data,
                use_container_width=True,
                height=500,
                hide_index=True
            )

        with tab3:
            st.subheader("Most Competitive Ridings")
            
            # Calculate competitiveness (difference between 1st and 2nd place)
            competitive_data = []
            for _, row in predicted_df.iterrows():
                # Get all predicted shares
                shares = {
                    'PC': row.get('Predicted_PC_Share', 0),
                    'NDP': row.get('Predicted_NDP_Share', 0),
                    'Liberal': row.get('Predicted_Liberal_Share', 0),
                    'Green': row.get('Predicted_Green_Share', 0),
                    'Other': row.get('Predicted_Other_Share', 0)
                }
                
                # Sort parties by vote share
                sorted_parties = sorted(shares.items(), key=lambda x: x[1], reverse=True)
                
                # Calculate margin between 1st and 2nd place
                if len(sorted_parties) >= 2:
                    margin = (sorted_parties[0][1] - sorted_parties[1][1]) * 100  # Convert to percentage points
                    competitive_data.append({
                        'Riding': row['Riding'],
                        'Leader': sorted_parties[0][0],
                        'Leader_Share': sorted_parties[0][1] * 100,
                        'Second': sorted_parties[1][0],
                        'Second_Share': sorted_parties[1][1] * 100,
                        'Margin': margin
                    })
            
            # Create DataFrame and sort by margin (closest races first)
            competitive_df = pd.DataFrame(competitive_data)
            competitive_df = competitive_df.sort_values('Margin', ascending=True)
            
            # Show top 20 closest races
            st.write("Top 20 Closest Ridings:")
            st.dataframe(
                competitive_df.head(20).style.format({
                    'Leader_Share': '{:.1f}%',
                    'Second_Share': '{:.1f}%',
                    'Margin': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Riding': 'Riding',
                    'Leader': st.column_config.TextColumn('Leading Party'),
                    'Leader_Share': st.column_config.NumberColumn('Leader %'),
                    'Second': st.column_config.TextColumn('2nd Place'),
                    'Second_Share': st.column_config.NumberColumn('2nd Place %'),
                    'Margin': st.column_config.NumberColumn('Margin')
                }
            )
            
            # Visualize distribution of competitiveness
            st.subheader("Competitiveness Distribution")
            
            fig = px.histogram(
                competitive_df,
                x='Margin',
                nbins=20,
                title='Distribution of Victory Margins Across All Ridings',
                labels={'Margin': 'Victory Margin (percentage points)'}
            )
            
            # Add vertical lines for reference
            fig.add_vline(x=5, line_dash="dash", line_color="orange", 
                         annotation_text="5% margin", annotation_position="top")
            fig.add_vline(x=10, line_dash="dash", line_color="red", 
                         annotation_text="10% margin", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)

    # Methodology page
    elif page == "Methodology":
        st.title("Ontario Election Predictor Methodology")
        
        st.header("How We Generate Our Projections")
        st.write("""
        Our model combines historical election results with current polling data to predict riding-level outcomes across Ontario. 
        Here's the step-by-step process we use:
        """)
        
        with st.expander("Historical Analysis"):
            st.markdown("""
            - Analyze the last two provincial elections (2018 and 2022)
            - Calculate vote shares by party
            - Uses machine learning to establish a seat model (e.g. winning 40% of the vote does not result in winning 40% of seats)
            """)
        
        with st.expander("Processing Current Polls"):
            st.markdown("""
            - Analyze polls from 2022 to present-day from a myriad of polling companies to fit to the seat model
            - Dashboard page allows filtering by date or polling company
            - Full list of polls can be downloaded on Dashboard
            """)
        
        with st.expander("Predicting Riding-Level Changes"):
            st.markdown("""
            Using machine learning, we estimate how provincial trends will play out in each riding by:
            - Applying historical swing patterns from 2018-2022
            - Ensuring riding projections match provincial polling averages
            - Please note that riding analysis does not allow for 'other' party projections. However, they are on riding map
            """)
        
        with st.expander("Monte Carlo: Simulating Uncertainty"):
            st.markdown("""
            We run 1,000 statistical simulations that account for:
            - Typical polling variability (±3% per party)
            - Potential regional variations
            - Different vote splitting scenarios
            
            This may result in precise variations in projections for the same inputs. However, these variations should not change the rounded outcome.
            
            This produces a range of likely outcomes with probabilities.
            """)
        
        with st.expander("Special Case: Star Candidates"):
            st.markdown("""
            When enabled, our model adjusts for high-profile candidates that often outperform the traditional party seat modelling. Under riding analysis, projections for all parties can be adjusted based on a possible 'star' candidate (e.g. increasing PC vote would also decrease other parties),
            """)
        
        st.header("Key Features of Our Model")
        st.table({
            "Aspect": [
                "Historical Data", 
                "Poll Aggregation",
                "Riding Projections",
                "Uncertainty",
                "Special Cases"
            ],
            "How We Handle It": [
                "Uses actual 2018 and 2022 results as baseline",
                "Weighted average prioritizing recent, large, reliable polls",
                "Machine learning applies provincial trends locally",
                "1,000 simulations provide confidence ranges",
                "Adjustable for star candidate effects"
            ]
        })
        
        st.header("Understanding the Outputs")
        st.markdown("""
        Our projections show:
        - **Most Likely Outcome**: The central seat projection
        - **Confidence Range**: Where results will likely fall (90% probability)
        - **Majority Probability**: Chance each party wins 63+ seats
        """)
        
        st.header("Limitations to Consider")
        st.markdown("""
        1. Smaller parties/independents are harder to predict precisely
        2. Unforeseen events (late campaign changes) aren't accounted for
        3. New candidates without history use party baselines
        """)
        
        st.info("""
        Note: These projections update automatically as new polling data becomes available.
        All calculations are based on publicly available election results and polling data.
        Please contact haydenghlai@gmail.com, if you have any suggestions or concerns.
        """)
        
else:
    st.warning("Please log in to access the application")