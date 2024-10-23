import datetime
import pandas as pd
import random as rd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def apply_counts(df: pd.DataFrame, count_col: str):
    """ Denormalise a dataframe with a 'Counts' column by
    multiplying that column by the count and dropping the 
    count_col. """
    feats = [c for c in df.columns if c != count_col]
    return pd.concat([
        pd.DataFrame([list(r[feats])] * r[count_col], columns=feats)
        for i, r in df.iterrows()
    ], ignore_index=True)


def main():
    
    dataframe1 = pd.read_excel(".\\Modified_Data_cat_personality.xlsx")
    dataframe = pd.read_excel(".\\Modified_Data_cat_personality.xlsx")

    sex = [0, 1]  
    age = [0, 1, 2, 3] 
    nombre = [1, 2, 3, 4, 5] 
    logement = [0, 1, 2, 3]  
    zone = [0, 1, 2]  
    ext = [1, 2, 3, 4, 5]  
    calme = [1, 2, 3, 4, 5]  
    effraye = [1, 2, 3, 4, 5] 
    intelligent = [1, 2, 3, 4, 5]  
    vigilant = [1, 2, 3, 4, 5]
    perseverant = [1, 2, 3, 4, 5]
    affectueux = [1, 2, 3, 4, 5]
    amical = [1, 2, 3, 4, 5]  
    solitaire = [1, 2, 3, 4, 5] 
    brutal = [1, 2, 3, 4, 5]  
    dominant = [1, 2, 3, 4, 5]  
    agressif = [1, 2, 3, 4, 5]  
    impulsif = [1, 2, 3, 4, 5]
    previsible = [1, 2, 3, 4, 5]  
    distrait = [1, 2, 3, 4, 5]  
    obs = [0, 1, 2, 3]  
    timide = [0, 1, 2, 3, 4]  
    abondance = [0, 1, 2, 3]  
    predOiseau = [0, 1, 2, 3, 4] 
    predMamm = [0, 1, 2, 3, 4]  

    Y = dataframe1["Race"]

    features = ["Sexe", "Logement", "Zone", "Ext", "Obs", "Timide", "Calme", "Effrayé", 
                    "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
                    "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif",
                     "Prévisible", "Distrait", "Abondance", "PredOiseau", "PredMamm"] 	

    # features = ["Sexe", "Logement", "Zone", "Ext", "Timide", "Calme", "Effrayé", 
    #                 "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
    #                 "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif",
    #                  "Prévisible", "Distrait", "Abondance", "PredOiseau", "PredMamm"]

    
    X = dataframe1[features]

    for i, j in dataframe1.iterrows():
        if j['Obs'] == 4:
            for k in range(3):
                dataframe1 = pd.concat([dataframe1, pd.DataFrame(j).T], ignore_index=True)
        if j['Obs'] == 3:
            for k in range(2):
                dataframe1 = pd.concat([dataframe1, pd.DataFrame(j).T], ignore_index=True)
        if j['Obs'] == 2:
            dataframe1 = pd.concat([dataframe1, pd.DataFrame(j).T], ignore_index=True)



    clf = DecisionTreeClassifier(criterion="entropy").fit(X,Y)  
    
    # fig, ax = plt.subplots(figsize=(7, 8))
    # f = tree.plot_tree(clf, ax=ax, fontsize=10, feature_names=features)
    # plt.show()
    column = []
    cnt = (dataframe[dataframe1.columns[0]].count())
    print(cnt)
    for i in range(0, 6):
        random_row = [
            cnt + i,
            datetime.datetime.now(),
            rd.choice(sex),
            rd.choice(age),
            rd.choice(nombre),
            rd.choice(logement),
            rd.choice(zone),
            rd.choice(ext),
            rd.choice(obs),
            rd.choice(timide),
            rd.choice(calme),
            rd.choice(effraye),
            rd.choice(intelligent),
            rd.choice(vigilant),  
            rd.choice(perseverant),  
            rd.choice(affectueux),  
            rd.choice(amical),
            rd.choice(solitaire),
            rd.choice(brutal),
            rd.choice(dominant),
            rd.choice(agressif),
            rd.choice(impulsif),
            rd.choice(previsible),
            rd.choice(distrait),
            rd.choice(abondance),
            rd.choice(predOiseau),
            rd.choice(predMamm),
            None
        ]
        
        column.append(random_row)
    all_features = ["Row.names", "Horodateur", "Sexe", "Age", "Nombre", "Logement", "Zone", "Ext", "Obs", "Timide", "Calme", "Effrayé", 
                    "Intelligent", "Vigilant", "Perséverant", "Affectueux", "Amical", 
                    "Solitaire", "Brutal", "Dominant", "Agressif", "Impulsif",
                     "Prévisible", "Distrait", "Abondance", "PredOiseau", "PredMamm", "Plus"]    
    new_instances_df = pd.DataFrame(column, columns=all_features)
    predictions = clf.predict(new_instances_df[features])

    new_instances_df["Race"] = predictions

    dataframe = pd.concat([dataframe, new_instances_df])

    dataframe.to_excel(r".\\Data_cat_personality_with_new_instances.xlsx", index=False)

if __name__ == "__main__":
    main()
