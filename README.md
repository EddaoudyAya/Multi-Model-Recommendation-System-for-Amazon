# Multi-Model Recommendation System for Amazon

## Project Overview

This project involves the design, implementation, and deployment of an **intelligent and modular recommendation system** for Amazon products. It integrates **four different recommendation approaches**, along with a **customer review sentiment analysis module**, all accessible through an **interactive web interface** built with Streamlit.

The system is designed to handle various real-world scenarios like cold-start problems, sparse user data, and the need for diverse recommendations.

---

## Key Features

- **Four Recommendation Models:**
  - **Popularity-Based Filtering** – For cold-start scenarios
  - **Content-Based Filtering** – Uses product descriptions and TF-IDF vectorization
  - **Collaborative Filtering** – Model-based using SVD matrix factorization
  - **Hybrid Approach** – Combines all three methods for robust recommendations

- **Sentiment Analysis:**
  - NLP-based classification of customer reviews
  - Bernoulli Naïve Bayes model achieving **70.43% accuracy**
  - Converts ratings (1-5 stars) into sentiment labels (-1, 0, 1)

- **Interactive Dashboard:**
  - Real-time product recommendations
  - Visual exploration of customer reviews and ratings
  - Multiple filtering options and result visualization

- **Scalable Architecture:**
  - Big Data processing with PySpark
  - MongoDB integration for data storage
  - Modular Python pipeline for data processing

---

## System Architecture

### Data Pipeline:
1. **Data Collection** – Amazon Reviews 2023 dataset from McAuley Lab
2. **Data Processing** – Cleaning, feature engineering, and text preprocessing
3. **Model Training** – Multiple recommendation algorithms and sentiment analysis
4. **Deployment** – Streamlit web application with interactive interface

### Technical Stack:
- **Data Processing:** PySpark, Pandas
- **Database:** MongoDB
- **ML Libraries:** Scikit-learn, Surprise, NLTK
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib, Tableau (for analytics dashboard)

---

## Model Performance

### Recommendation Models Evaluation (on 1,000 products sample):

| Method | Precision | Recall | Diversity |
|--------|-----------|--------|-----------|
| Content-Based | 47.4% | 47.4% | 0.986 |
| Collaborative | 0.4% | 0.4% | 0.991 |
| Popularity | 0.04% | 0.04% | 0.984 |

### Sentiment Analysis Models:

| Model | Accuracy | Status |
|-------|----------|--------|
| SVC (C=0.01) | 66.08% | - |
| Multinomial Naïve Bayes | 70.09% | - |
| **Bernoulli Naïve Bayes** | **70.43%** | **Selected** |

> **Note:** Collaborative and popularity-based models show better performance on full dataset (>30% precision) despite low scores on the 1,000-product sample due to data sparsity.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- MongoDB
- Java (for PySpark)

### Installation Steps

1. **Clone the repository:**
2. **Set up MongoDB:**

Install and start MongoDB service
Update connection URI in configuration files

3. **Download dataset**:

Obtain Amazon Reviews 2023 dataset from McAuley Lab
McAuley Lab Datasets: https://cseweb.ucsd.edu/~jmcauley/datasets.html

Place in data/raw/ directory

3. **Run data processing pipeline:**

```bash
python data_processing/data_cleaning.py
python data_processing/feature_generation.py
python data_processing/data_merge.py
```
Train models:

```bash
python models/sentiment_analysis.py
python models/collaborative_model_based.py
```
Launch web application:
```bash
streamlit run appstreamlit.py
```
## Methodology Details

### Data Processing:
- **Web Scraping Challenges:** Addressed Amazon's anti-bot protections by using McAuley Lab's curated datasets
- **Text Preprocessing:** HTML cleaning, stopword removal, stemming with PorterStemmer
- **Feature Engineering:** TF-IDF vectorization, review counting, sentiment scoring

### Recommendation Algorithms:
- **Content-Based:** Cosine similarity on product descriptions
- **Collaborative:** SVD matrix factorization with implicit feedback
- **Popularity:** Aggregate ratings and review counts
- **Hybrid:** Weighted combination of all methods

### Evaluation Metrics:
- **Precision@K** – Relevance of top-K recommendations
- **Recall@K** – Coverage of relevant items
- **Diversity** – Variety in recommendations (1 - average similarity)

---

## Results & Insights

- **Content-based filtering** performs best on small datasets with rich product descriptions
- **Collaborative filtering** requires substantial user interaction data to be effective
- **Sentiment analysis** adds valuable contextual understanding beyond star ratings
- **Hybrid approach** provides the most robust coverage across different scenarios

---

## Usage Examples

### Demo Video
[![System Demo](https://img.shields.io/badge/Watch_Demo-FF6B6B?style=for-the-badge&logo=video&logoColor=white)](demo.mp4)

### Through Web Interface:
1. **Select a recommendation model** from the sidebar
2. **Search and select a product** from the dropdown
3. **Click "Generate Recommendations"** to see similar products
4. **View detailed results** with images, ratings, and prices

---
## Feedback

**Have fun exploring the system!** If you have any questions, suggestions for improvement, or would like to contribute to enhancing the recommendation algorithms, feel free to reach out! 


