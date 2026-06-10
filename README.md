# 🇳🇬 Fake News Detection System

A machine learning system that detects whether a news article is **fake or real**, 
with a focus on Nigerian and international news. Built using NLP and classical ML 
classifiers, deployed via a Streamlit web application.

 **Live Demo:** [Try it here](https://fakenewsdetectionsystemproject-qe4naupznvzbaukkfwg3uz.streamlit.app/)

---

##  Problem Statement

The spread of fake news has become a major societal challenge, particularly in Nigeria 
where misinformation spreads rapidly across social media platforms. This project builds 
an automated classifier to detect fake news articles, supporting fact-checking efforts 
and combating disinformation.

---

##  Objectives

- Collect and merge multiple fake and real news datasets
- Preprocess and clean text data using NLP techniques
- Train and evaluate machine learning classifiers
- Deploy a user-friendly web application for real-time prediction

---

##  Dataset

Four datasets were merged into a single corpus (~60,000 rows):

| Dataset | Source | Label |
|---|---|---|
| ISOT Fake News | University of Victoria | Fake (1) |
| ISOT True News | University of Victoria | Real (0) |
| Nigeria 2019 Newsfeed | Nigerian news sources | Real (0) |
| Nigerian Fake News | Scraped from Dubawa & Africa Check | Fake (1) |

The Nigerian fake news dataset consists of **350 articles** scraped from 
[Dubawa](https://dubawa.org) and [Africa Check](https://africacheck.org) — 
two leading Nigerian fact-checking platforms.

---

##  Project Pipeline

1. Introduction & Problem Statement
2. Importing Dependencies
3. Dataset Overview & Observations
4. Data Cleaning
5. Data Normalization
6. Data Visualization
7. Correlation & Heatmap
8. Data Preprocessing
9. Feature Selection & Target Assignment
10. Encoding, Feature Scaling & Standardization
11. Train Test Split (80/20)
12. Model Selection
13. Model Training
14. Model Prediction
15. Model Evaluation
16. Feature Importance
17. Saving the Model
18. Model Application

---

##  Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 97.13% |
| LinearSVC (SVM) | 97.99%  |

**LinearSVC was selected as the final model** based on higher accuracy and fewer 
misclassifications on both classes.

---

##  Model Evaluation

### Logistic Regression Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
|---|---|---|
| **Actual Real (0)** | 8262 | 88  |
| **Actual Fake (1)** | 290  | 4539  |

### SVM Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
|---|---|---|
| **Actual Real (0)** | 8271  | 79  |
| **Actual Fake (1)** | 186  | 4643  |

SVM produced fewer misclassifications on both classes.

---

## 🔍 Feature Importance

Top words indicating **Fake News:** `via`, `image`, `gop`, `hillary`, `obama`, `breitbart`, `fox`

Top words indicating **Real News:** `reuters`, `said`, `nigeria`, `buhari`, `nigerian`, `washington`

---

##  Limitations

- The model is biased toward **US political fake news** due to the dominance of the 
  ISOT dataset in training
- Nigerian fake news data was limited (~350 articles, ~14 unique after deduplication), 
  reducing the model's ability to detect Nigerian-specific fake news
- Nigerian-specific terms (buhari, lagos, nigerian) overlap with real news vocabulary, 
  causing the model to misclassify some Nigerian fake news as real
- A larger and more diverse Nigerian fake news dataset would significantly improve performance

---

##  Streamlit Web Application

The system is deployed as a web app where users can paste any news article and 
receive an instant prediction with a confidence score.

 **Live Demo:** [fakenewsdetectionsystemproject.streamlit.app](https://fakenewsdetectionsystemproject-qe4naupznvzbaukkfwg3uz.streamlit.app/)

**Features:**
- Real-time fake/real classification
- Confidence score display
- Clean dark-themed UI

---

##  Tech Stack

- **Language:** Python 3.13
- **Libraries:** Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Joblib, Streamlit
- **ML Models:** Logistic Regression, LinearSVC
- **NLP:** TF-IDF Vectorization, Regex, NLTK (Stopwords, Lemmatization)
- **Environment:** Conda (fakenews), Jupyter Notebook, VS Code
- **Deployment:** Streamlit Cloud

---

##  Repository Structure

```
Fake_news_detection_system_project/
├── News_project.ipynb                  # Main Jupyter Notebook (full pipeline)
├── app.py                              # Streamlit web application
├── fake_news_svm_model.joblib          # Saved SVM model
├── fake_news_logreg_model.joblib       # Saved Logistic Regression model
├── News_tfidf_vectorizer.joblib        # Saved TF-IDF vectorizer
├── Fake.csv                            # ISOT fake news dataset
├── True.csv                            # ISOT true news dataset
├── Nigeria2019_Newsfeed.csv            # Nigerian real news dataset
├── nigerian_fake_news.csv              # Nigerian fake news (Dubawa & Africa Check)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

##  How to Run Locally

**Requirements:** Python 3.7+

```bash
# Clone repo
git clone https://github.com/Ayomhi1234/Fake_news_detection_system_project.git
cd Fake_news_detection_system_project

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

##  Future Improvements

- Source a larger Nigerian fake news dataset for better local context
- Add Nigerian language support (Yoruba, Hausa, Igbo)
- Integrate real-time news feed monitoring
- Implement ensemble methods (Random Forest, Gradient Boosting)
- Add explainability with LIME/SHAP

---

##  Author

**Ayomide Somide** — Data Scientist | NYSC | Pharmaceutical Industry  
GitHub: [Ayomhi1234](https://github.com/Ayomhi1234)

---

## Acknowledgments

- **Dubawa** & **Africa Check** — Nigerian fact-checking platforms (fake news data source)
- **University of Victoria** — ISOT Fake News Dataset
- **Libraries:** Scikit-learn, Streamlit, Pandas, NLTK
