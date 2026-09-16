# Movie Recommender

A content-based movie recommendation system built with Python and the TMDB 5000 Movie Dataset. The project uses movie metadata such as genres, keywords, cast, crew, and overview to generate relevant recommendations for a given movie title.

## Project Overview

This project analyzes movie descriptions and attributes, combines them into a single text field called `tags`, and then converts those tags into vectors using `CountVectorizer`. After that, it calculates cosine similarity between movies and recommends the most similar titles.

The system is implemented in the notebook located at:

- `Data/data_preproces.ipynb`

## Dataset

The project uses the following TMDB files:

- `Data/tmdb_5000_movies.csv`
- `Data/tmdb_5000_credits.csv`

Dataset Link:- `https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata`

These datasets contain movie information such as:

- title
- genres
- keywords
- overview
- cast
- crew
- release date
- popularity
- vote statistics

## How It Works

1. Load the movie and credit datasets.
2. Merge both datasets on the `title` column.
3. Clean the dataset by selecting relevant columns and removing missing values.
4. Convert nested JSON-like columns such as `genres`, `keywords`, `cast`, and `crew` into cleaner Python lists.
5. Build a combined `tags` column from:
   - overview
   - genres
   - keywords
   - cast
   - director
6. Convert all text to lowercase and apply stemming using PorterStemmer.
7. Vectorize the text using `CountVectorizer`.
8. Compute cosine similarity between movie vectors.
9. Recommend the top 3 movies similar to the selected movie.

## Recommendation Logic

The notebook defines a function similar to:

```python
recommend("Avatar")
```

This returns the top movies most similar to Avatar based on shared content features.

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- scikit-learn
- Jupyter Notebook

## Setup Instructions

1. Clone or open the project folder.
2. Make sure Python is installed.
3. Install the required libraries:

```bash
pip install pandas numpy nltk scikit-learn
```

4. Open the notebook in Jupyter or VS Code.
5. Run the cells in sequence.

## Example

```python
# Example call
recommend("Avatar")
```

You will get a list of similar movies ranked by cosine similarity.

## Notes

- This is a content-based filtering approach rather than collaborative filtering.
- It works well when the user wants recommendations based on movie metadata and description similarity.
- The model can be extended with additional features like user ratings, popularity-based filtering, or hybrid recommendation systems.

## File Structure

```text
Movie_Recommender/
├── Data/
│   ├── data_preproces.ipynb
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── Readme.md
└── .gitignore (if added later)
```

## License

This project is intended for educational and learning purposes.
