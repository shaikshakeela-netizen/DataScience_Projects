# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
/* Set overall background color */
.stApp {
    background-color: #eaf2f8;
}

/* Main title */
h1 {
    color: #2c3e50;
    text-align: center;
    background-color: #d6eaf8;
    padding: 12px;
    border-radius: 8px;
}

/* Subheadings */
h2, h3, h4 {
    color: #2c3e50;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #d4efdf !important;
    color: #1b2631;
}

/* Buttons */
.stButton>button {
    background-color: #1abc9c;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #16a085;
    color: white;
}

/* DataFrame styling */
.dataframe {
    border: 1px solid #ccc;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #d6eaf8;
    border-radius: 8px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    color: #2c3e50;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App Title
# -----------------------------
st.markdown("<h1>💳 Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Options
# -----------------------------
st.sidebar.header("📂 Upload & Model Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
model_type = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
sample_size = st.sidebar.slider("Select Sample Size for Training", 5000, 50000, 10000, step=5000)

# -----------------------------
# Load and Cache Data
# -----------------------------
if uploaded_file:
    @st.cache_data
    def load_data(file):
        data = pd.read_csv(file)
        return data

    data = load_data(uploaded_file)

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "⚙️ Model Training", "🔮 Prediction"])

    # -----------------------------
    # Tab 1: EDA
    # -----------------------------
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(data.head())
        st.write("Shape:", data.shape)
        st.write("Fraud Count:", int(data['Class'].sum()))
        st.write("Non-Fraud Count:", len(data) - int(data['Class'].sum()))

        # Class Distribution
        st.subheader("Transaction Class Distribution")
        fig = px.histogram(data, x="Class", color="Class",
                           color_discrete_map={0: "green", 1: "red"},
                           title="Fraud vs Non-Fraud Transactions")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # Tab 2: Model Training
    # -----------------------------
    with tab2:
        st.subheader(f"Model Training: {model_type}")

        X = data.drop('Class', axis=1)
        y = data['Class']

        # Take a smaller subset for faster execution
        data_sample = data.sample(sample_size, random_state=42)
        X_sample = data_sample.drop('Class', axis=1)
        y_sample = data_sample['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)

        # Handle Imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y_sample)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        with st.spinner("🚀 Training the model... please wait"):
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # Evaluation
        y_pred = model.predict(X_test)
        st.subheader("📈 Model Evaluation")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.text("Classification Report:")
        report = classification_report(y_test, y_pred)
        st.text(report)

        # Store model & scaler for prediction tab
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["columns"] = X_sample.columns.tolist()

    # -----------------------------
    # Tab 3: Prediction
    # -----------------------------
    with tab3:
        st.subheader("🔍 Predict a Transaction")

        if "model" not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
        else:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            columns = st.session_state["columns"]

            st.write("Enter transaction details below:")
            input_data = {}
            for col in columns:
                val = st.number_input(f"{col}", value=float(data[col].mean()))
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])[columns]
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]

                if prediction == 1:
                    st.error("⚠️ This transaction is FRAUDULENT!")
                else:
                    st.success("✅ This transaction is NON-FRAUDULENT!")
