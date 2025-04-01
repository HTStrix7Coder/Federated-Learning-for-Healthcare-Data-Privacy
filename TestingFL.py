import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved global model
model_path = r"C:\Users\harin\OneDrive\Federated-Learning-for-Healthcare-Data-Privacy-main\FLModel\federated_healthcare_model.h5"
global_model = tf.keras.models.load_model(model_path)
print("Global model loaded successfully!")

# Load the scaler fitted on the training data
scaler_path = scaler_path = r"C:\Users\harin\OneDrive\Federated-Learning-for-Healthcare-Data-Privacy-main\FLModel\scaler.pkl"
scaler = joblib.load(scaler_path)
print("Scaler loaded successfully!")

expected_features = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
    'sex_Female', 'sex_Male',
    'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'fbs_False', 'fbs_True','restecg_lv hypertrophy','restecg_normal','restecg_st-t abnormality',
    'exang_False', 'exang_True',
    'slope_downsloping', 'slope_flat', 'slope_upsloping',
    'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
]

# Data to Predict
test_data = {
    'age': [45, 60],
    'trestbps': [130, 150],
    'chol': [230, 260],
    'thalch': [150, 170],
    'oldpeak': [2.3, 1.8],
    'ca': [0, 2],
    # One-hot encoded categorical features:
    'sex_Female': [0, 1],
    'sex_Male': [1, 0],
    'cp_asymptomatic': [0, 1],
    'cp_atypical angina': [1, 0],
    'cp_non-anginal': [0, 0],
    'cp_typical angina': [0, 0],
    'fbs_False': [1, 1],
    'fbs_True': [0, 0],
    'restecg_lv hypertrophy': [1, 0],
    'restecg_normal': [0, 1],
    'restecg_st-t abnormality': [0, 0],
    'exang_False': [1, 0],
    'exang_True': [0, 1],
    'slope_downsloping': [1, 0],
    'slope_flat': [0, 1],
    'slope_upsloping': [0, 0],
    'thal_fixed defect': [0, 1],
    'thal_normal': [1, 0],
    'thal_reversable defect': [0, 0]
}

test_df = pd.DataFrame(test_data)

# Ensure the DataFrame columns are in the exact order expected by the model
test_df = test_df[expected_features]

# Standardize the features.
# Using the scaler fitted on the training data.
test_data_scaled = scaler.transform(test_df)

# Make predictions using the global model
predictions = global_model.predict(test_data_scaled)
predictions = (predictions > 0.5).astype(int)
print("Predictions for test data:")
print(predictions)

true_labels = np.array([0, 1])

# Performance metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()