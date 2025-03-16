import numpy as np
import tensorflow as tf
import pandas as pd
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if TensorFlow is using GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU!")
else:
    print("TensorFlow is not using GPU.")

# Ignore future warnings
warnings.simplefilter('ignore', FutureWarning)

# Load the dataset
data = pd.read_csv('MLProject/heart_disease_uci.csv')

# Identify categorical columns
categorical_columns = [
    'sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'
]

# Apply One-Hot Encoding
data = pd.get_dummies(data, columns=categorical_columns)

# Ensure all columns that should be numeric are numeric
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values (optional, depending on your data handling strategy)
#data.dropna(inplace=True)

# Check for any remaining non-numeric columns and convert them if necessary
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Column {col} is not numeric. Converting to numeric.")
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any NaN values again after conversion
data.dropna(inplace=True)

# Exclude specific features from training
features_to_exclude = ['id','dataset_Cleveland','dataset_Hungary','dataset_Switzerland','dataset_VA Long Beach']
data = data.drop(columns=features_to_exclude)

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

scaler_path = "C:/Users/harin/Downloads/AICode/MLProject/scaler.pkl"
joblib.dump(scaler, scaler_path)
print("Scaler saved successfully!")

# Create multiple Hospital clients with their own local data
NUM_CLIENTS = 5
client_data_splits = np.array_split(np.column_stack((X, y)), NUM_CLIENTS)

# Generate client data
def generate_client_data(client_data):
    x = client_data[:, :-1]
    y = client_data[:, -1]
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

clients_data = [generate_client_data(client_data) for client_data in client_data_splits]

# Define a simple MLP model for healthcare predictions
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Federated training simulation: Simulate federated averaging manually
def federated_averaging(global_model, client_models):
    # Get the weights from each client model
    client_weights = [model.get_weights() for model in client_models]
    # Calculate the average weights
    avg_weights = []
    for weights_list in zip(*client_weights):  # Zip to iterate through each layer's weights
        avg_weights.append(np.mean(weights_list, axis=0))
    # Set the global model's weights to the average of the client models
    global_model.set_weights(avg_weights)

# Initialize the global model
global_model = create_model()

# Simulate federated training
for round_num in range(5):  # 5 training rounds
    client_models = []
    # Train local models for each client
    # Here the model is trained locally so no data leaks
    for client_data in clients_data:
        client_model = create_model()  # Create a new model for each client
        print(f"Training model for client {round_num}...")
        client_model.fit(client_data, epochs=1)
        print("Client data remains local and is not shared.")
        client_models.append(client_model)
    # After training, perform federated averaging to update the global model
    federated_averaging(global_model, client_models)
    print(f"Round {round_num + 1} completed")

# Save the final federated model
global_model.save("C:/Users/harin/Downloads/AICode/MLProject/federated_healthcare_model.h5")
print("Federated model saved successfully!")