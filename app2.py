import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import plotly.express as px

# -----------------------------------------------------------
# 1. Page Configuration & Setup
# -----------------------------------------------------------
st.set_page_config(page_title="Global Student Expense Predictor", page_icon="🌍", layout="wide")

# Hardcoded Credentials for Login
USERNAME = "admin"
PASSWORD = "password123"

# Dictionary of Currencies and Approximate Exchange Rates (Base: 1 USD)
CURRENCIES = {
    "United States (USD)": {"symbol": "$", "rate": 1.0},
    "India (INR)": {"symbol": "₹", "rate": 83.0},
    "Europe (EUR)": {"symbol": "€", "rate": 0.92},
    "United Kingdom (GBP)": {"symbol": "£", "rate": 0.79},
    "Australia (AUD)": {"symbol": "A$", "rate": 1.52},
    "Canada (CAD)": {"symbol": "C$", "rate": 1.36},
    "Japan (JPY)": {"symbol": "¥", "rate": 150.0}
}

# -----------------------------------------------------------
# 2. Login Page
# -----------------------------------------------------------
def login_page():
    st.title("🔒 Login to Global Expense Predictor")
    st.markdown("Please enter your credentials to access the dashboard.")
    
    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username_input == USERNAME and password_input == PASSWORD:
                st.session_state['logged_in'] = True
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Incorrect username or password. Please try again.")

# -----------------------------------------------------------
# 3. Model Training (Cached based on Currency Rate)
# -----------------------------------------------------------
@st.cache_resource
def get_trained_model(exchange_rate):
    # Load dataset
    try:
        df = pd.read_csv('student_spending.csv')
    except FileNotFoundError:
        st.warning("⚠️ 'student_spending.csv' not found. Using generated data.")
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(18, 25, 100),
            "gender": np.random.choice(["Male", "Female", "Non-binary"], 100),
            "year_in_school": np.random.choice(["Freshman", "Sophomore", "Junior", "Senior"], 100),
            "major": np.random.choice(["Computer Science", "Biology", "Engineering", "Business", "Art"], 100),
            "monthly_income": np.random.randint(500, 2000, 100),
            "financial_aid": np.random.randint(0, 1000, 100),
            "tuition": np.random.randint(3000, 6000, 100),
            "housing": np.random.randint(500, 1500, 100),
            "food": np.random.randint(200, 600, 100),
            "transportation": np.random.randint(50, 200, 100),
            "books_supplies": np.random.randint(50, 300, 100),
            "entertainment": np.random.randint(20, 150, 100),
            "personal_care": np.random.randint(20, 100, 100),
            "technology": np.random.randint(50, 300, 100),
            "health_wellness": np.random.randint(20, 150, 100),
            "miscellaneous": np.random.randint(10, 100, 100),
            "preferred_payment_method": np.random.choice(["Cash", "Credit/Debit Card", "Mobile Payment App"], 100)
        })

    # --- ADD MONTH COLUMN ---
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    if 'month' not in df.columns:
        np.random.seed(42)
        df['month'] = np.random.choice(months, size=len(df))

    # --- CONVERT CURRENCY BASED ON SELECTED RATE ---
    monetary_cols = [
        "monthly_income", "financial_aid", "tuition", "housing", "food", 
        "transportation", "books_supplies", "entertainment", "personal_care", 
        "technology", "health_wellness", "miscellaneous"
    ]
    
    # Ensure we only multiply base USD values (assuming if avg tuition < 20000 it's USD)
    if df['tuition'].mean() < 20000:
        for col in monetary_cols:
            if col in df.columns:
                df[col] = df[col] * exchange_rate

    # Define features and target
    expense_cols = ["tuition", "housing", "food", "transportation", "books_supplies",
                    "entertainment", "personal_care", "technology", "health_wellness", "miscellaneous"]
    
    y = df[expense_cols].sum(axis=1)  # Target is total expense
    
    feature_cols = [
        "age", "gender", "year_in_school", "major", "month", 
        "monthly_income", "financial_aid", "tuition", "housing", "food", 
        "transportation", "books_supplies", "entertainment", "personal_care", 
        "technology", "health_wellness", "miscellaneous", "preferred_payment_method"
    ]
    
    X = df[feature_cols].copy()
    
    # Encode categorical features
    encoders = {}
    categorical_cols = ["gender", "year_in_school", "major", "month", "preferred_payment_method"]
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y)
    
    return encoders, scaler, model, feature_cols, months

# -----------------------------------------------------------
# 4. Main Application
# -----------------------------------------------------------
def main_app():
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_country = st.selectbox("Select Country/Currency", list(CURRENCIES.keys()), index=1) # Default to India
        
        # Get Symbol and Rate
        sym = CURRENCIES[selected_country]["symbol"]
        rate = CURRENCIES[selected_country]["rate"]
        
        st.divider()
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # Retrieve model mapped to the selected currency
    encoders, scaler, model, feature_cols, month_list = get_trained_model(rate)

    st.title(f"🎓 Student Expense Predictor ({selected_country})")
    st.markdown(f"Enter your details below to predict your total expected spending for the month in **{selected_country} ({sym})**.")

    # --- User Inputs ---
    with st.form("user_input_form"):
        st.markdown("### 👤 Demographics & Academics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=50, value=20)
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
        with col2:
            year_in_school = st.selectbox("Year in School", ["Freshman", "Sophomore", "Junior", "Senior"])
            major = st.selectbox("Major", ["Computer Science", "Biology", "Economics", "Psychology", "Engineering", "Business"])
        with col3:
            month = st.selectbox("Month", month_list)
            preferred_payment_method = st.selectbox("Preferred Payment Method", ["Cash", "Credit/Debit Card", "Mobile Payment App"])

        st.markdown(f"### 💰 Monthly Income & Financial Aid (in {sym})")
        col4, col5 = st.columns(2)
        with col4:
            monthly_income = st.number_input(f"Monthly Income ({sym})", min_value=0, value=int(1000 * rate), step=int(50 * rate))
        with col5:
            financial_aid = st.number_input(f"Financial Aid ({sym})", min_value=0, value=int(500 * rate), step=int(50 * rate))

        st.markdown(f"### 💸 Estimated Expenses (in {sym})")
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            tuition = st.number_input(f"Tuition ({sym})", min_value=0, value=int(4800 * rate), step=int(100 * rate))
            housing = st.number_input(f"Housing ({sym})", min_value=0, value=int(700 * rate), step=int(50 * rate))
            food = st.number_input(f"Food ({sym})", min_value=0, value=int(300 * rate), step=int(20 * rate))
        with e_col2:
            transportation = st.number_input(f"Transportation ({sym})", min_value=0, value=int(100 * rate), step=int(10 * rate))
            books_supplies = st.number_input(f"Books & Supplies ({sym})", min_value=0, value=int(180 * rate), step=int(10 * rate))
            entertainment = st.number_input(f"Entertainment ({sym})", min_value=0, value=int(100 * rate), step=int(10 * rate))
        with e_col3:
            personal_care = st.number_input(f"Personal Care ({sym})", min_value=0, value=int(50 * rate), step=int(10 * rate))
            technology = st.number_input(f"Technology ({sym})", min_value=0, value=int(100 * rate), step=int(10 * rate))
        with e_col4:
            health_wellness = st.number_input(f"Health & Wellness ({sym})", min_value=0, value=int(70 * rate), step=int(10 * rate))
            miscellaneous = st.number_input(f"Miscellaneous ({sym})", min_value=0, value=int(50 * rate), step=int(10 * rate))

        submitted = st.form_submit_button("Predict Total Spending")

    # --- Processing & Prediction ---
    if submitted:
        input_dict = {
            "age": age, "gender": gender, "year_in_school": year_in_school, "major": major, "month": month,
            "monthly_income": monthly_income, "financial_aid": financial_aid, 
            "tuition": tuition, "housing": housing, "food": food, 
            "transportation": transportation, "books_supplies": books_supplies, 
            "entertainment": entertainment, "personal_care": personal_care, 
            "technology": technology, "health_wellness": health_wellness, 
            "miscellaneous": miscellaneous, "preferred_payment_method": preferred_payment_method
        }
        
        df_new = pd.DataFrame([input_dict])
        
        for col in ["gender", "year_in_school", "major", "month", "preferred_payment_method"]:
            if df_new[col].iloc[0] not in encoders[col].classes_:
                df_new[col] = encoders[col].classes_[0]
            df_new[col] = encoders[col].transform(df_new[col])
            
        df_new = df_new[feature_cols]
        X_new_scaled = scaler.transform(df_new)
        prediction = model.predict(X_new_scaled)[0]
        
        st.divider()
        st.success(f"### 🎯 Predicted Total Monthly Spending: {sym}{prediction:,.2f}")
        
        # --- Generate Graphs ---
        st.markdown("### 📊 Your Spending Visualized")
        
        expenses_dict = {
            "Tuition": tuition, "Housing": housing, "Food": food, 
            "Transportation": transportation, "Books & Supplies": books_supplies, 
            "Entertainment": entertainment, "Personal Care": personal_care, 
            "Technology": technology, "Health & Wellness": health_wellness, 
            "Miscellaneous": miscellaneous
        }
        
        expenses_df = pd.DataFrame(list(expenses_dict.items()), columns=['Category', 'Amount'])
        expenses_df = expenses_df[expenses_df['Amount'] > 0] 
        
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            fig_pie = px.pie(expenses_df, values='Amount', names='Category', title='Expense Distribution', hole=0.4)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with graph_col2:
            fig_bar = px.bar(expenses_df.sort_values(by='Amount', ascending=True), 
                             x='Amount', y='Category', orientation='h', 
                             title=f'Expenses by Category ({sym})', color='Amount', color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------
# 5. App Routing
# -----------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()