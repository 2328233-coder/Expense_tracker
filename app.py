import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import plotly.express as px

# -----------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="Student Spending Predictor", page_icon="🎓", layout="wide")

# -----------------------------------------------------------
# 2. Auto-Train Model (Fixes the Column Mismatch)
# -----------------------------------------------------------
@st.cache_resource
def get_trained_model():
    try:
        df = pd.read_csv('student_spending.csv')
    except FileNotFoundError:
        st.warning("⚠️ 'student_spending.csv' not found. Using generated data so the app can run.")
        data = {
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
            "books_supplies": np.random.randint(100, 400, 100),
            "entertainment": np.random.randint(50, 300, 100),
            "personal_care": np.random.randint(20, 100, 100),
            "technology": np.random.randint(50, 200, 100),
            "health_wellness": np.random.randint(30, 150, 100),
            "miscellaneous": np.random.randint(20, 100, 100),
            "preferred_payment_method": np.random.choice(["Cash", "Credit/Debit Card", "Mobile Payment App"], 100)
        }
        df = pd.DataFrame(data)

    # Clean up 'Unnamed: 0' if it exists in the CSV
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 1. Preprocess categorical columns
    encoders = {}
    label_cols = ["gender", "year_in_school", "major", "preferred_payment_method"]
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # 2. Calculate Target Variable (Total Expenses)
    expense_cols = ["tuition", "housing", "food", "transportation", "books_supplies", 
                    "entertainment", "personal_care", "technology", "health_wellness", "miscellaneous"]
    y = df[expense_cols].sum(axis=1)
    
    # 3. Select EXACTLY 17 Features (No Unnamed: 0)
    feature_cols = [
        "age", "gender", "year_in_school", "major", 
        "monthly_income", "financial_aid", "tuition", "housing", "food", 
        "transportation", "books_supplies", "entertainment", "personal_care", 
        "technology", "health_wellness", "miscellaneous", "preferred_payment_method"
    ]
    
    X = df[feature_cols]
    
    # 4. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Train XGBoost
    model = XGBRegressor(random_state=42)
    model.fit(X_scaled, y)
    
    return encoders, scaler, model, feature_cols

# Load the auto-trained model
encoders, scaler, model, feature_cols = get_trained_model()

# -----------------------------------------------------------
# 3. Build the Streamlit User Interface
# -----------------------------------------------------------
st.title("🎓 Interactive Student Spending Predictor")
st.markdown("Enter your demographics and monthly expenses below to predict your overall spending.")

# --- Layout: Sidebar for Demographics, Main for Expenses ---
st.sidebar.header("📋 Student Information")
age = st.sidebar.number_input("Age", min_value=16, max_value=100, value=21)

gender_options = list(encoders["gender"].classes_)
gender = st.sidebar.selectbox("Gender", options=gender_options)

year_options = list(encoders["year_in_school"].classes_)
year_in_school = st.sidebar.selectbox("Year in School", options=year_options)

major_options = list(encoders["major"].classes_)
major = st.sidebar.selectbox("Major", options=major_options)

payment_options = list(encoders["preferred_payment_method"].classes_)
preferred_payment_method = st.sidebar.selectbox("Preferred Payment Method", options=payment_options)

monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, value=1200)
financial_aid = st.sidebar.number_input("Financial Aid ($)", min_value=0, value=500)

# --- Main Area: Expenses ---
st.header("💸 Monthly Expenses Breakdown")
col1, col2, col3 = st.columns(3)

with col1:
    tuition = st.number_input("Tuition ($)", min_value=0, value=4500)
    housing = st.number_input("Housing ($)", min_value=0, value=800)
    food = st.number_input("Food ($)", min_value=0, value=350)
    transportation = st.number_input("Transportation ($)", min_value=0, value=100)

with col2:
    books_supplies = st.number_input("Books & Supplies ($)", min_value=0, value=200)
    entertainment = st.number_input("Entertainment ($)", min_value=0, value=150)
    personal_care = st.number_input("Personal Care ($)", min_value=0, value=50)

with col3:
    technology = st.number_input("Technology ($)", min_value=0, value=100)
    health_wellness = st.number_input("Health & Wellness ($)", min_value=0, value=80)
    miscellaneous = st.number_input("Miscellaneous ($)", min_value=0, value=70)

st.markdown("---")

# -----------------------------------------------------------
# 4. Prediction & Visualization Logic
# -----------------------------------------------------------
if st.button("Predict Total Spending & Generate Graphs", type="primary", use_container_width=True):
    
    # --- Data Prep for Model (EXACTLY 17 columns) ---
    input_data = {
        "age": [age],
        "gender": [gender],
        "year_in_school": [year_in_school],
        "major": [major],
        "monthly_income": [monthly_income],
        "financial_aid": [financial_aid],
        "tuition": [tuition],
        "housing": [housing],
        "food": [food],
        "transportation": [transportation],
        "books_supplies": [books_supplies],
        "entertainment": [entertainment],
        "personal_care": [personal_care],
        "technology": [technology],
        "health_wellness": [health_wellness],
        "miscellaneous": [miscellaneous],
        "preferred_payment_method": [preferred_payment_method]
    }
    
    df_new = pd.DataFrame(input_data)
    
    # Encode categorical columns
    label_cols = ["gender", "year_in_school", "major", "preferred_payment_method"]
    for col in label_cols:
        df_new[col] = encoders[col].transform(df_new[col])
            
    # Ensure order matches exactly
    df_new = df_new[feature_cols]
    
    # Scale and Predict
    X_new_scaled = scaler.transform(df_new)
    prediction = model.predict(X_new_scaled)[0]
    
    # --- Display Prediction Results ---
    st.success(f"### 🎯 Predicted Total Monthly Spending: ${prediction:,.2f}")
    
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
        fig_pie = px.pie(
            expenses_df, 
            values='Amount', 
            names='Category', 
            title='Expense Distribution',
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with graph_col2:
        fig_bar = px.bar(
            expenses_df.sort_values(by='Amount', ascending=True), 
            x='Amount', 
            y='Category', 
            orientation='h', 
            title='Expenses by Category',
            color='Amount',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)