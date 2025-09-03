import joblib
import pandas as pd

# Load the saved model:
model = joblib.load("model/category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input(" Enter review title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    text = input(" Enter review text: ")
    if text.lower() == "exit":
        print("Exiting...")
        break