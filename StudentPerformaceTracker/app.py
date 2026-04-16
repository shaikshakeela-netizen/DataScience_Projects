import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Student Performance Tracker", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
    <style>
    /* System font with white text everywhere */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #ffffff !important; /* all text white */
        font-style: italic;  /* italic style */
    }

    /* Full-page animated gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #0a0f1c, #1c1c3a, #2c5364);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        color: #ffffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111111, #2a2a2a);
        color: #ffffff !important;
        font-weight: 500;
        font-style: italic;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-style: italic !important;
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #00f260, #0575e6); /* bright header */
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        font-size: 40px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7);
        margin-bottom: 25px;
        animation: fadeInDown 1s ease-in-out;
        color: #ffffff !important; /* white text */
        font-style: italic;
    }

    /* Upload box */
    .upload-box {
        border: 3px dashed #00f260; /* bright green border */
        border-radius: 20px;
        background: rgba(0,0,0,0.3);
        padding: 35px;
        text-align: center;
        color: #ffffff !important; /* white text */
        font-style: italic;
        transition: all 0.4s ease;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
    }

    .upload-box:hover {
        transform: scale(1.03);
        background: rgba(0,0,0,0.5);
        border-color: #0575e6;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.7);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #f5af19, #f12711);
        color: #ffffff !important; /* all button text white */
        font-weight: 600;
        font-style: italic;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        background: linear-gradient(90deg, #f12711, #f5af19);
        box-shadow: 0 6px 20px rgba(0,0,0,0.7);
        color: #ffffff !important;
    }

    /* Inputs and selects */
    .stTextInput input, .stSelectbox select, .stFileUploader input {
        color: #ffffff !important; /* white input text */
        font-weight: 500 !important;
        font-style: italic !important;
        background-color: rgba(0,0,0,0.5) !important; /* dark semi-transparent box */
        border-radius: 8px;
    }

    /* Animations */
    @keyframes gradientFlow {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-30px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    /* File uploader text and box */
    .stFileUploader label, 
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        color: #ffffff !important;          /* White label text */
        background-color: rgba(255,255,255,0.1) !important; /* Semi-transparent box */
        border: 2px dashed #ffffff !important; /* White dashed border */
        border-radius: 12px;
        padding: 15px;
        font-style: italic;                 /* Optional italic */
        font-weight: 500;
        text-align: center;
    }

    /* Hover effect */
    .stFileUploader div[data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255,255,255,0.2) !important;
        border-color: #00f260 !important; /* Bright green highlight on hover */
    }
    /* Make all input/select/file uploader labels white */
    label, .css-1kyxreq, .css-1d391kg {
        color: white !important;
        font-weight: 500;
        font-style: italic;
    }
    /* Make selectbox label text white while keeping green background */
    label[data-baseweb="select"] {
        color: white !important;   /* white text */
        font-weight: 600;          /* bold text */
        font-style: italic;        /* italic style */
        background-color: #00f260 !important; /* green background */
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    </style>
""", unsafe_allow_html=True)






# -------------------- HEADER --------------------
st.markdown('<div class="main-header"><h1>🎓 Student Performance Tracker</h1></div>', unsafe_allow_html=True)

# -------------------- SIDEBAR MENU --------------------
menu = ["🏠 Home", "📊 EDA Analysis", "🤖 ML Prediction"]
choice = st.sidebar.radio("Navigation", menu)

# -------------------- FILE UPLOAD --------------------

uploaded_file = st.file_uploader("Upload your student performance dataset (CSV)", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# -------------------- OVERVIEW --------------------
st.subheader("Dataset Overview")
st.dataframe(df.head())
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# -------------------- HOME --------------------
if choice == "🏠 Home":
    st.markdown("""
    Welcome to the **Student Performance Tracker**  
    This AI-powered tool:
    - Automatically analyzes **any student dataset**
    - Performs **EDA (graphs, correlations, trends)**
    - Builds a **machine learning model**
    - Shows **predictions and feature importance**
    """)
    st.info("👉 Use the sidebar to explore EDA or ML Prediction modules.")

# -------------------- EDA --------------------
elif choice == "📊 EDA Analysis":
    st.subheader("Exploratory Data Analysis")

    st.write("### Data Summary")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Data Types")
    st.write(df.dtypes)

    # Visualizations
    st.write("### Distribution Plot")
    num_col = st.selectbox("Select a numerical column", num_cols)
    fig1 = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}", color_discrete_sequence=['#6a11cb'])
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### Correlation Heatmap")
    if len(num_cols) > 1:
        fig2, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)

    st.write("### Pairplot (sampled)")
    if len(num_cols) > 1:
        sample_df = df.sample(min(200, len(df)))
        st.pyplot(sns.pairplot(sample_df[num_cols]).fig)

# -------------------- MACHINE LEARNING --------------------
elif choice == "🤖 ML Prediction":
    st.subheader("Machine Learning Prediction")

    target_col = st.selectbox("Select target column (what you want to predict)", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Detect model type
    model_type = "Classifier" if len(np.unique(y)) < 10 else "Regressor"

    if model_type == "Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        st.success(f"Trained a Regression Model — R² Score: **{score:.3f}**")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        st.success(f"Trained a Classification Model — Accuracy: **{score*100:.2f}%**")

    # Feature importance
    st.write("### Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig3 = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                  title='Feature Importance', color='Importance',
                  color_continuous_scale='blues')
    st.plotly_chart(fig3, use_container_width=True)

    # Try a custom prediction
    st.write("### Try Custom Prediction")
    user_input = {}
    for col in X.columns:
        val = st.text_input(f"{col}", value=str(X[col].iloc[0]))
        user_input[col] = [float(val)] if col in num_cols else [int(val)]

    input_df = pd.DataFrame(user_input)
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Value: **{prediction}**")
