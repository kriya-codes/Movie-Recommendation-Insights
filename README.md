# 🎬 Movie Recommendation Insights Using Python

A content-based movie recommendation system built with Python that suggests similar movies based on genre information using TF-IDF Vectorization and Cosine Similarity.

## 📌 Project Overview

Finding movies that match our interests can be difficult when a dataset contains thousands of titles. This project explores movie metadata and recommends similar movies based on their genre information.

The system takes a movie title as input and returns a ranked list of similar movies.

## 🎯 Objectives

- Explore and preprocess movie dataset information.
- Analyze movie genres and identify patterns.
- Convert genre information into numerical features using TF-IDF.
- Calculate movie similarity using Cosine Similarity.
- Generate recommendations for a selected movie.

## ✨ Features

- Movie dataset exploration and preprocessing
- Genre-based content representation
- Content-based movie recommendations
- Similarity-based ranking
- Release-year tie-breaking where implemented

## 🛠️ Technologies Used

- **Python** — core programming language
- **Pandas** — dataset loading and manipulation
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **Jupyter Notebook** — implementation and experimentation

## ⚙️ How It Works

1. Load the movie dataset.
2. Inspect and prepare movie metadata.
3. Process genre information.
4. Convert genre text into TF-IDF vectors.
5. Calculate cosine similarity between the selected movie and other movies.
6. Rank similar movies and display recommendations.

## 📂 Project Files

- `Movie_Recommendation_Insights_Updated.ipynb` — main project notebook
- `movie_recommendation_updated.py` — supporting Python implementation, if included
- `movies.csv` — dataset, if included and permitted for redistribution

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kriya-codes/Movie-Recommendation-Insights.git
cd Movie-Recommendation-Insights
```

### 2. Install dependencies

Install the required libraries in your Python environment:

```bash
pip install pandas scikit-learn jupyter
```

Check the notebook imports and install any additional dependencies it requires.

### 3. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. Ensure the dataset is available at the path expected by the code.

## 📊 Dataset

The dataset contains movie identifiers, titles, and genre information. Recommendations are generated using the available metadata.

The dataset does not include user ratings or viewing history, so this project focuses on content-based recommendations rather than collaborative filtering.

## ⚠️ Limitations

- Recommendations depend mainly on genre information.
- The system does not use personal viewing history or user ratings.
- Similar genre combinations may produce similar recommendations.

## 🔮 Future Improvements

- Include plot descriptions and keywords for richer recommendations.
- Develop an interactive web interface.
- Explore additional movie metadata where reliable data is available.
- Add user preferences for more personalized recommendations.

## 👩‍💻 **Author**

**Kriya Parmar**

**GitHub**: @kriya-codes

---
Developed a project to explore movie data and implement a content-based recommendation approach.
