import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Spam Email Detector Ready!")
print("Type 'exit' to stop.\n")

while True:
    msg = input("Enter message: ")

    if msg.lower() == "exit":
        break

    # Convert text to numbers
    msg_vector = vectorizer.transform([msg])

    # Predict
    prediction = model.predict(msg_vector)[0]

    if prediction == 1:
        print("🚫 SPAM message\n")
    else:
        print("✅ NOT spam\n")