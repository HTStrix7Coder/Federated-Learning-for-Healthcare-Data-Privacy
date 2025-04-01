import numpy as np
import tensorflow as tf
import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if TensorFlow is using GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU!")
else:
    print("TensorFlow is not using GPU.")
warnings.simplefilter('ignore', FutureWarning)

# Load the heart disease dataset
data = pd.read_csv('heart_disease_uci.csv')

# Identify categorical columns in the Dataset
categorical_columns = [
    'sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'
]

# Apply One-Hot Encoding
data = pd.get_dummies(data, columns=categorical_columns)

# Ensure all columns that should be numeric are numeric
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Convert any remaining non-numeric columns if present
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(inplace=True)

# Exclude specific irrelevant features
features_to_exclude = ['id','dataset_Cleveland','dataset_Hungary','dataset_Switzerland','dataset_VA Long Beach']
data = data.drop(columns=features_to_exclude)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
scaler_path = r"C:\Users\harin\OneDrive\Federated-Learning-for-Healthcare-Data-Privacy-main\FLModel\scaler.pkl"
joblib.dump(scaler, scaler_path)
print("Scaler saved successfully!")

# Split data among clients
NUM_CLIENTS = 5
client_data_splits = np.array_split(np.column_stack((X, y)), NUM_CLIENTS)

# Function to generate client data as tf.data.Dataset
def generate_client_data(client_data):
    x = client_data[:, :-1]
    y = client_data[:, -1]
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

clients_data = [generate_client_data(client_data) for client_data in client_data_splits]

# Define model creation function
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Federated averaging function
def federated_averaging(global_model, client_models):
    client_weights = [model.get_weights() for model in client_models]
    avg_weights = []
    for weights in zip(*client_weights):
        avg_weights.append(np.mean(weights, axis=0))
    global_model.set_weights(avg_weights)

# Initialize global model
global_model = create_model()

# Federated training with tracking of global accuracy
federated_accuracies = []

for round_num in range(5):  # 5 training rounds
    client_models = []
    round_accuracies = []
    
    for client_idx, client_data in enumerate(clients_data):
        client_model = create_model()
        client_model.set_weights(global_model.get_weights())  # start from global weights
        
        print(f"Training model for client {client_idx + 1}, round {round_num + 1}...")
        client_model.fit(client_data, epochs=5, verbose=0)  # Train for more epochs
        
        # Evaluate on client data
        loss, acc = client_model.evaluate(client_data, verbose=0)
        print(f"Client {client_idx + 1} accuracy: {acc * 100:.2f}%")
        round_accuracies.append(acc * 100)
        
        client_models.append(client_model)
    
    # Federated averaging
    federated_averaging(global_model, client_models)
    
    # Evaluate global model on combined dataset (just for tracking)
    global_loss, global_acc = global_model.evaluate(tf.data.Dataset.from_tensor_slices((X, y)).batch(10), verbose=0)
    print(f"\nRound {round_num + 1} completed. Global model accuracy: {global_acc * 100:.2f}%\n")
    
    federated_accuracies.append(global_acc * 100)

# Save the final federated model
global_model.save(r"C:\Users\harin\OneDrive\Federated-Learning-for-Healthcare-Data-Privacy-main\FLModel\federated_healthcare_model.h5")
print("Federated model saved successfully!")

# Plot global accuracy per round
plt.plot(range(1, 6), federated_accuracies, marker='o')
plt.title('Global Model Accuracy Over Federated Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.show()
