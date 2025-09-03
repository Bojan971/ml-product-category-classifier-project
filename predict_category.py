import joblib
import pandas as pd

# Load the saved model:
model = joblib.load("model/category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input(" Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break


    # Compute title and word length:
    title_length = len(title)
    word_count = len(title.split())

    # Create a DataFrame from input:
    user_input = pd.DataFrame([{
        "Product Title": title,
        "title_length": title_length,
        "word_count": word_count
        }])

    # Predict category:
    prediction = model.predict(user_input)[0]
    print(f" Predicted category: {prediction}\n" + "-" * 40)