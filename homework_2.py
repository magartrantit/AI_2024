from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from reteaua import Network, CrossEntropyCost, RELu
from collections import Counter


def preprocess_data_with_smote(file_path):

    data = pd.read_excel(file_path)
    
    data = data.drop(columns=['Horodateur', 'Row.names'])
    
    X = data.drop(columns=['Race'])  
    y = data['Race']
    
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)
    
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    val_data = [(x, y) for x, y in zip(X_val, y_val)]
    
    class_counts = Counter(y_resampled)
    mapped_counts = {label_encoder.inverse_transform([k])[0]: v for k, v in class_counts.items()}
    
    print("Distribuție clase după SMOTE:", mapped_counts)
    
    return train_data, val_data, label_encoder


def train_cat_classifier(training_data, validation_data, input_size, hidden_sizes, output_size, epochs, batch_size, lr, reg_param):
    cost = CrossEntropyCost()
    activation = RELu()
    network = Network(input_size, hidden_sizes, output_size, cost, activation)
    network.train(training_data, epochs, batch_size, lr, reg_param, validation_data)
    return network, None  

if __name__ == "__main__":
    # Path către dataset
    dataset_path = ".\\Modified_Data_cat_personality.xlsx"

    # Parametrii rețelei
    hidden_layer_sizes = [64, 32]
    epochs = 10
    batch_size = 16
    lr = 0.005
    reg_param = 0.1  

    # Preprocesare date cu SMOTE
    train_data, val_data, _ = preprocess_data_with_smote(dataset_path)

    # Determinarea dimensiunilor pentru rețea
    num_features = len(train_data[0][0])  # Determinare automată din date
    num_classes = len(np.unique([y for _, y in train_data]))

    # Antrenare rețea
    trained_network, encoder = train_cat_classifier(
        training_data=train_data,
        validation_data=val_data,
        input_size=num_features,
        hidden_sizes=hidden_layer_sizes,
        output_size=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg_param=reg_param
    )
    
    print("Model training complete.")
