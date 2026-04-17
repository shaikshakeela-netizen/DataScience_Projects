
# 🌦 WeatherSense – Weather Forecast Data Analysis Dashboard

An interactive **Streamlit dashboard** for weather data analysis, visualization, and short-term temperature forecasting using Machine Learning.

---

## 🚀 Project Overview

**WeatherSense** is a data analytics application that enables users to:

* 📊 Perform **Exploratory Data Analysis (EDA)**
* 🎯 Apply **dynamic filters (date, temperature, precipitation)**
* 📈 Visualize trends using **interactive charts**
* 🤖 Predict future temperatures using **Linear Regression**
* 💡 Generate meaningful **data insights**

This project demonstrates strong skills in **Data Analysis, Visualization, and ML-based Forecasting**.

---

## 🛠 Tech Stack

* **Programming:** Python
* **Framework:** Streamlit
* **Libraries:**

  * Pandas, NumPy
  * Matplotlib, Seaborn, Plotly
  * Scikit-learn

---

## 📂 Features

### 🔹 1. Data Upload

* Upload any weather dataset (CSV format)
* Automatic column cleaning and preprocessing

### 🔹 2. Data Filtering

* 📅 Date range selection (restricted up to 2016)
* 🌡 Temperature range slider
* ☔ Precipitation type filter

### 🔹 3. Exploratory Data Analysis (EDA)

* Line Chart (Time Series)
* Histogram (Distribution)
* Boxplot (Outliers)
* Scatter Plot (Relationships)

### 🔹 4. Forecasting (ML)

* Uses **Linear Regression**
* Predicts temperature for next **1–7 days**
* Based on previous values (lag feature)

### 🔹 5. Insights Generation

* Average, max, and min temperature
* Correlation heatmap between variables

---

## 📸 Dashboard Preview

*Add screenshots here (recommended)*
Example:

```
/screenshots/dashboard.png
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/weathersense.git
cd weathersense
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
weathersense/
│
├── app.py                  # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── screenshots/           # Dashboard images
```

---

## 📊 Sample Dataset Requirements

Your CSV should include at least:

* `Formatted Date`
* `Temperature (C)`

Optional:

* `Apparent Temperature (C)`
* `Precip Type`

---

## 🤖 Machine Learning Logic

* Feature Engineering:

  * Uses previous temperature (`Prev_Value`) as input
* Model:

  * Linear Regression
* Output:

  * Future predictions for selected days

---

## 📌 Key Highlights

* ✅ Interactive UI with **custom dark theme**
* ✅ Real-time filtering & visualization
* ✅ End-to-end **data pipeline (EDA → ML → Insights)**
* ✅ Beginner-friendly and production-ready design

---

## 🔮 Future Improvements

* Add advanced ML models (ARIMA, LSTM)
* Real-time API integration (live weather data)
* Deployment using Streamlit Cloud / AWS
* Add user authentication

---

## 👩‍💻 Author

**Shakeela Shaik**
📧 [shaikshakeela004@gmail.com](mailto:shaikshakeela004@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/shakeela-shaik-b75b06326/)

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share with others

