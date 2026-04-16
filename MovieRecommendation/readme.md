# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![NLP](https://img.shields.io/badge/NLP-CountVectorizer-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Cosine%20Similarity-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

Choosing what to watch can be overwhelming with thousands of available movies.
This project builds a **content-based Movie Recommendation System** that suggests similar movies based on user selection.

The application uses **Natural Language Processing (NLP)** techniques to analyze movie metadata and delivers personalized recommendations through an **interactive Netflix-inspired UI**.

---

## 🎯 Key Objectives

* Recommend similar movies based on content
* Utilize **NLP techniques** for feature extraction
* Build a fast and scalable recommendation engine
* Provide an engaging **Netflix-style user experience**

---

## 🛠️ Tech Stack

| Category        | Tools Used               |
| --------------- | ------------------------ |
| Programming     | Python                   |
| Data Handling   | Pandas                   |
| NLP             | CountVectorizer          |
| Similarity      | Cosine Similarity        |
| UI / Deployment | Streamlit                |
| Dataset         | TMDB 5000 Movies Dataset |

---

## ⚙️ Features

✅ Content-based movie recommendations
✅ Uses **genres + keywords + overview** for similarity
✅ Fast similarity computation using vectorization
✅ Netflix-style dark UI 🎥
✅ Interactive movie selection dropdown
✅ Displays **Top 5 personalized recommendations**

---

## 📊 How It Works

```text
Movie Metadata → Text Processing → Feature Extraction (CountVectorizer)
→ Vector Representation → Cosine Similarity → Top 5 Recommendations
```

---

## 🧠 Recommendation Logic

* Combine:

  * Genres
  * Keywords
  * Overview

* Convert text into numerical vectors using **CountVectorizer**

* Compute similarity using **cosine similarity**

* Recommend movies with highest similarity scores


## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/movie-recommendation-system.git

# Navigate to project folder
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```



---

## 💡 Key Highlights

* Real-world application of **Recommendation Systems**
* Demonstrates **NLP + ML concepts clearly**
* Clean and engaging **Netflix-like UI design**
* Efficient similarity computation
* Strong portfolio project for **Data Analyst / ML roles**

---

## 🔮 Future Improvements

* Add collaborative filtering
* Use advanced NLP (TF-IDF, Word2Vec, BERT)
* Integrate movie posters using TMDB API
* Deploy on cloud (AWS / Streamlit Cloud)
* Add user login & personalized recommendations

---

## 👩‍💻 Author

**Shakeela Shaik**
📊 Data Analyst | Machine Learning Enthusiast

🔗 LinkedIn: https://www.linkedin.com/in/shakeela-shaik-b75b06326/

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
