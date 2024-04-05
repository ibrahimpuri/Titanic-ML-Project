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