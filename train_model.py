print("Script started")
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("cleaned_data.csv")

X = data.drop("CLASS", axis=1)
y = data["CLASS"]


#Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set size:", X_train)

# Train the model (you can choose your model)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test,  y_pred))

dump(model, 'diabetes_model.joblib')
