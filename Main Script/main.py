import pandas as pd

# Load the training data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few row of the training data
train_data.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
print(train_data.isnull().sum())

# Survival Rate
print(f"Survival Rate: {train_data['Survived'].mean()*100:.2f}%")

# Visualise the survival rate by gender
sns.barplot(x='sex', y='Survived', data=train_data)
plt.show()

# Visualise the survival rate by passenger rate
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Fill missing age values with the median age
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])

# Convert 'Sex' to numerical values
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = encoder.transform(test_data['Sex'])

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

from sklearn.ensemble import RandomForestClassifier

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the training data
predictions = model.predict(X_test)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the training data for evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train_split, y_train_split)

# Predict on the validation data
val_predictions = model.predict(X_val_split)

# Calculate the accuracy
accuracy = accuracy_score(y_val_split, val_predictions)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the training data for evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train_split, y_train_split)

# Predict on the validation data
val_predictions = model.predict(X_val_split)

# Calculate the accuracy
accuracy = accuracy_score(y_val_split, val_predictions)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_val_split, rf_predictions)
recall = recall_score(y_val_split, rf_predictions)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

from sklearn.metrics import roc_auc_score, roc_curve

rf_predictions_proba = rf_model.predict_proba(X_val_split)

# AUC score
auc = roc_auc_score(y_val_split, rf_predictions_proba[:, 1])
print(f"AUC: {auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val_split, rf_predictions_proba[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

# Create a base model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_