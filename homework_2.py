import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random as rd
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_excel(r".\\Modified_Data_cat_personality.xlsx")


    X = data[[
        'Sexe', 'Age', 'Nombre', 'Logement', 'Zone', 'Ext', 'Obs', 
        'Timide', 'Calme', 'Effrayé', 'Intelligent', 'Vigilant', 'Perséverant', 
        'Affectueux', 'Amical', 'Solitaire', 'Brutal', 'Dominant', 'Agressif', 
        'Impulsif', 'Prévisible', 'Distrait', 'Abondance', 'PredOiseau', 'PredMamm'
    ]]
    y = data['Race'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Dimensiunea setului de antrenare (X_train):", X_train.shape)
    print("Dimensiunea setului de testare (X_test):", X_test.shape)
    print("Dimensiunea etichetelor de antrenare (y_train):", y_train.shape)
    print("Dimensiunea etichetelor de testare (y_test):", y_test.shape)
        
        
    
if __name__ == "__main__":
    main()