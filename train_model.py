import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data=data[['v1', 'v2']]
data.columns=['label', 'message']
#VERY IMPORTANT FIX
data['message']=data['message'].astype(str)
print(data.head())
print(data.shape)
# Convert labels to numbers
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")

