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

from sklearn.ensemble import GradientBoostingClassifier

# Initialize and fit the Gradient Boosting classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)

# Predict on the training set
gb_predictions = gb_clf.predict(X_val_split)

# Evaluate the model
gb_accuracy = accuracy_score(y_val_split, gb_predictions)
print(f"Validation Accuracy of Gradient Boosting: {gb_accuracy*100:.2f}%")

from xgboost import XGBClassifier

# Initialize and fit the XGBoost classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_split, y_train_split)

# Predict on the validation set
xgb_predictions = xgb_clf.predict(X_val_split)

# Evaluate the model
xgb_accuracy = accuracy_score(y_val_split, xgb_predictions)
print(f"Validation Accuracy of XGBoost: {xgb_accuracy*100:.2f}%")

# Feature importance from the Gradient Boosting model
feature_importance = gb_clf.feature_importances_

# Plot
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance')
plt.show()