
# 🧬 Breast Cancer Detection – ML Streamlit App

An end-to-end **Machine Learning + Streamlit application** for predicting breast cancer diagnosis (Benign vs Malignant) with interactive UI, visual analytics, and explainability.

---

## 🚀 Project Overview

This project is a **multi-phase ML system** designed to:

* 🎯 Predict breast cancer diagnosis using classification models
* 📊 Visualize feature importance and model performance
* 🧠 Provide explainability with multiple ML algorithms
* 📁 Support bulk predictions via CSV upload
* 📄 Generate downloadable PDF reports

It demonstrates strong skills in **Machine Learning, Data Analysis, UI/UX, and Deployment-ready systems**.

---

## 🛠 Tech Stack

* **Language:** Python
* **Framework:** Streamlit
* **ML Models:**

  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors
* **Libraries:**

  * Pandas, NumPy
  * Scikit-learn
  * Matplotlib, Seaborn, Plotly
  * Joblib

---

## 📂 Project Structure

```
breast-cancer-ml-app/
│
├── app.py                    # Main Streamlit application
├── src/
│   ├── data.py              # Data loading & preprocessing
│   ├── model.py             # Model training & evaluation
│   ├── ui.py                # UI styling & components
│   ├── api.py               # FastAPI integration (optional)
│
├── utils/
│   ├── pdf_report.py        # PDF generation for predictions
│
├── models/                  # Saved models
├── data/                    # Dataset
├── requirements.txt
└── README.md
```

---

## ⚙️ Features (Phase-wise)

### 🔹 Phase 1: ML Foundation

* Model training (Logistic Regression)
* Accuracy evaluation
* Confusion matrix visualization

### 🔹 Phase 2: UX & Visualization

* Feature importance analysis
* Interactive charts
* PDF report generation

### 🔹 Phase 3: Interactivity

* 🔍 Single prediction (manual input)
* 📁 Bulk prediction via CSV upload
* 🕒 Prediction history tracking

### 🔹 Phase 4: Explainability

* Compare multiple models:

  * Logistic Regression
  * Random Forest
  * KNN
* Feature importance visualization
* Performance metrics comparison

### 🔹 Phase 5: Deployment (Optional)

* FastAPI integration
* REST API support for predictions

---

## 📊 Sample Dataset

* Based on **Breast Cancer dataset**
* Features include:

  * Radius, Texture, Perimeter, Area, Smoothness, etc.

---

## 🧠 Machine Learning Workflow

1. Data Loading & Cleaning
2. Feature Scaling
3. Model Training
4. Evaluation (Accuracy, Confusion Matrix)
5. Prediction (Single + Bulk)
6. Model Saving (Joblib)

---

## ▶️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-ml-app.git
cd breast-cancer-ml-app
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📄 PDF Report Feature

* Generates prediction reports including:

  * Diagnosis result
  * Confidence score
  * Model accuracy
  * Input features

---

## 📸 Screenshots (Add Here)

```
/screenshots/home.png
/screenshots/prediction.png
/screenshots/visualization.png
```

---

## 🔮 Future Enhancements

* Add Deep Learning models
* Deploy on Streamlit Cloud / AWS
* Add real-time API integration
* Improve UI animations & responsiveness

---

## 👩‍💻 Author

**Shakeela Shaik**
📧 [shaikshakeela004@gmail.com](mailto:shaikshakeela004@gmail.com)
🔗 https://www.linkedin.com/in/shakeela-shaik-b75b06326/

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share it

---

