import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# Load CSV file:
df = pd.read_csv("data/products.csv")

# Drop all rows with missing values:
df = df.dropna()

# Removal of leading/trailing spaces and normalization to the same writing:
df.columns = df.columns.str.strip()

# Mapping Category Label (target variables):
merge_map = {
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridges"
}

# Value replacement in the Category Label column:
df["Category Label"] = df["Category Label"].replace(merge_map)

# Create a new columns:
df = df.copy()

df["title_length"] = df["Product Title"].apply(len)
df["word_count"] = df["Product Title"].apply(lambda x: len(x.split()))

# Drop columns which are not useful for modeling:
df.columns = df.columns.str.strip()

df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', 'Number_of_Views', 'Merchant Rating', 'Listing Date'])

# Features and label:
X = df[["Product Title", "title_length", "word_count"]]
y = df["Category Label"]

# Define preprocessing: TF-IDF for text, MinMaxScaler for numeric feature:
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title"),
        ("word", MinMaxScaler(), ["word_count"]),
        ("length", MinMaxScaler(), ["title_length"])
    ]
)

# Define pipeline with the best model (e.g. Suport Vector Machine):
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset:
pipeline.fit(X, y)

# Save the model to a file:
joblib.dump(pipeline, "model/category_model.pkl")