import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample bug data (for demonstration purposes)
bug_data = {
    'description': ['App crashes when clicking on the "Submit" button.',
                    'User authentication not working properly.',
                    'Homepage layout is broken on mobile devices.',
                    'Slow performance when loading large datasets.'],
    'priority': ['High', 'High', 'Medium', 'Low']  # Priority label for each bug
}

# Convert data to DataFrame
bugs_df = pd.DataFrame(bug_data)

# Preprocessing: Convert priority labels to numerical values
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
bugs_df['priority'] = bugs_df['priority'].map(priority_mapping)

# Feature extraction: Convert text descriptions into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(bugs_df['description'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, bugs_df['priority'], test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)