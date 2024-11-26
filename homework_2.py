import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from reteaua import Network, CrossEntropyCost, RELu

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_excel(file_path)
    
    # Drop irrelevant columns (Timestamp and rowname)
    data = data.drop(columns=['Horodateur', 'Row.names'])
    
    # Separate features and target
    X = data.drop(columns=['Race'])  # Ensure 'Race' is the correct target column
    y = data['Race']
    
    # Encode categorical features
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Prepare data in required format
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    val_data = [(x, y) for x, y in zip(X_val, y_val)]
    
    return train_data, val_data, label_encoder

def train_cat_classifier(file_path, input_size, hidden_sizes, output_size, epochs=30, mini_batch_size=16, learning_rate=0.01):
    # Preprocess the data
    train_data, val_data, label_encoder = preprocess_data(file_path)
    
    # Initialize the network
    network = Network(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        cost=CrossEntropyCost,
        activation=RELu
    )
    
    network.train(
    training_data=train_data,
    epochs=epochs,
    batch_size=mini_batch_size,  
    lr=learning_rate,            
    validation_data=val_data
    )

    
    return network, label_encoder

if __name__ == "__main__":
    dataset_path = ".\\Modified_Data_cat_personality.xlsx"
    # Load the dataset for feature analysis
    dataset = pd.read_excel(dataset_path)

    # Drop irrelevant columns (Timestamp and rowname)
    dataset = dataset.drop(columns=['Horodateur', 'Row.names'])

    # Determine features and target dynamically
    features = dataset.columns.drop('Race')
    target = 'Race'
    num_features = len(features)
    num_classes = dataset[target].nunique()

    
    for column in features:
        distinct_values = dataset[column].value_counts()
    
    # Train the neural network
    hidden_layer_sizes = [64, 32]  # Example configuration for hidden layers
    trained_network, encoder = train_cat_classifier(
        file_path=dataset_path,
        input_size=num_features,
        hidden_sizes=hidden_layer_sizes,
        output_size=num_classes
    )
    
    print("Model training complete.")
