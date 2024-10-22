import pandas as pd
import random as rd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



def main():
    
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")

    sex = [0, 1]  
    age = [0, 1, 2, 3] 
    nombre = [1, 2, 3, 4, 5] 
    logement = [0, 1, 2, 3]  
    zone = [0, 1, 2]  
    ext = [1, 2, 3, 4, 5]  
    calme = [1, 2, 3, 4, 5]  
    effraye = [1, 2, 3, 4, 5] 
    intelligent = [1, 2, 3, 4, 5]  
    amical = [1, 2, 3, 4, 5]  
    solitaire = [1, 2, 3, 4, 5] 
    brutal = [1, 2, 3, 4, 5]  
    dominant = [1, 2, 3, 4, 5]  
    agressif = [1, 2, 3, 4, 5]  
    impulsif = [1, 2, 3, 4, 5]  #
    previsible = [1, 2, 3, 4, 5]  
    distrait = [1, 2, 3, 4, 5]  
    obs = [0, 1, 2, 3]  
    timide = [0, 1, 2, 3, 4]  
    abondance = [0, 1, 2, 3]  
    predOiseau = [0, 1, 2, 3, 4] 
    predMamm = [0, 1, 2, 3, 4]  

    Y = dataframe1["Race"]

    X = dataframe1[["Sexe", "Logement", "Zone", "Ext", "Obs", "Timide", "Calme", "Effrayé", 
                    "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
                    "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif", "Prévisible"]]
    features = ["Sexe", "Logement", "Zone", "Ext", "Obs", "Timide", "Calme", "Effrayé", 
                    "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
                    "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif", "Prévisible"]

    clf = DecisionTreeClassifier(criterion="entropy").fit(X,Y)  
    
    
    # fig, ax = plt.subplots(figsize=(7, 8))
    # f = tree.plot_tree(clf, ax=ax, fontsize=10, feature_names=features)
    # plt.show()
    column = []

    for i in range(0, 6):
        random_row = [
            rd.choice(sex),
            rd.choice(logement),
            rd.choice(zone),
            rd.choice(ext),
            rd.choice(obs),
            rd.choice(timide),
            rd.choice(calme),
            rd.choice(effraye),
            rd.choice(intelligent),
            rd.choice([1, 2, 3, 4, 5]),  
            rd.choice([1, 2, 3, 4, 5]),  
            rd.choice([1, 2, 3, 4, 5]),  
            rd.choice(amical),
            rd.choice(solitaire),
            rd.choice(brutal),
            rd.choice(dominant),
            rd.choice(agressif),
            rd.choice(impulsif),
            rd.choice(previsible)
        ]
        
        column.append(random_row)

    new_instances_df = pd.DataFrame(column, columns=["Sexe", "Logement", "Zone", "Ext", "Obs", "Timide", "Calme", "Effrayé", 
                                                     "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
                                                     "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif", "Prévisible"])

    predictions = clf.predict(new_instances_df)

    new_instances_df["Predicted_Race"] = predictions

    print(new_instances_df)

if __name__ == "__main__":
    main()