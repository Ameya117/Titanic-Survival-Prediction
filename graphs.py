import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Function to classify and return accuracy and CV score
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    cv_score = np.mean(cross_val_score(model, X, y, cv=5))
    return accuracy, cv_score

# Load your data
train_data = pd.read_csv('data/train.csv')

# Preprocess the data
# Drop columns that are not needed or are problematic
train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin'])

# Fill missing values
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())

# Convert categorical columns to numerical values
label_encoders = {}
for column in ['Sex', 'Embarked']:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])

X = train_data.drop(columns=["Survived"], axis=1)
y = train_data["Survived"]

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

# Collect results
results = []
for model_name, model in models.items():
    accuracy, cv_score = classify(model, X, y)
    results.append({"Model": model_name, "Accuracy": accuracy, "CV Score": cv_score})

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="CV Score", data=results_df, palette="viridis")
plt.title("Model CV Score Comparison")
plt.show()

# Display the results table
print(results_df)