import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
DATA_PATH = "/mnt/data/files/movies.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. BASIC CLEANING
# -----------------------------
df = df.drop_duplicates(subset="movieId").copy()
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["genres"] = df["genres"].fillna("").astype(str).str.strip()

# Remove rows without usable title/genre information
# Keep '(no genres listed)' out of the recommendation feature.
df = df[df["title"] != ""].copy()

# -----------------------------
# 3. EXTRACT RELEASE YEAR
# -----------------------------
def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else pd.NA


df["year"] = df["title"].apply(extract_year).astype("Int64")

# -----------------------------
# 4. PREPARE GENRE FEATURES
# -----------------------------
df["genre_features"] = (
    df["genres"]
    .str.replace("|", " ", regex=False)
    .str.replace("(no genres listed)", "", regex=False)
    .str.lower()
    .str.strip()
)

# -----------------------------
# 5. TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
tfidf_matrix = vectorizer.fit_transform(df["genre_features"])

# -----------------------------
# 6. COSINE SIMILARITY
# -----------------------------


# -----------------------------
# 7. RECOMMENDATION FUNCTION
# -----------------------------
def recommend_movies(movie_title, n=10):
    """Return n movies most similar to the requested movie title."""
    if not isinstance(movie_title, str) or not movie_title.strip():
        raise ValueError("Please enter a movie title.")

    query = movie_title.strip()

    # Exact title match first
    matches = df.index[df["title"].str.lower() == query.lower()].tolist()

    # Partial match if exact title is not found
    if not matches:
        partial = df[df["title"].str.lower().str.contains(query.lower(), regex=False)]
        if partial.empty:
            raise ValueError(f"Movie '{movie_title}' was not found in the dataset.")
        matches = [partial.index[0]]

    movie_idx = matches[0]
    row_position = df.index.get_loc(movie_idx)

    query_vector = tfidf_matrix[row_position]
    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()
    query_year = df.iloc[row_position]["year"]

    # Primary ranking: cosine similarity.
    # Tie-breaker: release-year proximity.
    candidates = []
    for pos, score in enumerate(scores):
        if pos == row_position:
            continue

        candidate_year = df.iloc[pos]["year"]
        if pd.notna(query_year) and pd.notna(candidate_year):
            year_distance = abs(int(candidate_year) - int(query_year))
        else:
            year_distance = float("inf")

        candidates.append((pos, float(score), year_distance))

    candidates.sort(key=lambda x: (-x[1], x[2]))

    results = []
    for pos, score, _ in candidates[:n]:
        results.append({
            "title": df.iloc[pos]["title"],
            "genres": df.iloc[pos]["genres"],
            "year": df.iloc[pos]["year"],
            "similarity": round(score, 4)
        })

    return pd.DataFrame(results)


# -----------------------------
# 8. BASIC DATASET SUMMARY
# -----------------------------
print("Dataset shape:", df.shape)
print("Columns:", ["movieId", "title", "genres", "year"])
print("Missing values:\n", df[["movieId", "title", "genres"]].isna().sum())
print("Duplicate movie IDs:", df["movieId"].duplicated().sum())

# -----------------------------
# 9. SAMPLE TESTS
# -----------------------------
if __name__ == "__main__":
    test_movies = ["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)"]

    for movie in test_movies:
        print("\\n" + "=" * 70)
        print("Recommendations for:", movie)
        try:
            print(recommend_movies(movie, n=10).to_string(index=False))
        except ValueError as e:
            print(e)
