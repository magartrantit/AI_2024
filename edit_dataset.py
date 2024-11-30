import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random as rd

def to_number(attribute, dataframe):
    enc = LabelEncoder()
    enc.fit(dataframe[attribute])
    dataframe[attribute] = enc.transform(dataframe[attribute])

def handle_missing_values(dataframe):

    sum_values = dataframe.loc[dataframe['Abondance'] != 'NSP', 'Abondance'].astype(int).sum()
    count_values = dataframe.loc[dataframe['Abondance'] != 'NSP', 'Abondance'].astype(int).count()
    final_number = round(sum_values / count_values, 2)
    dataframe['Abondance'] = dataframe['Abondance'].replace('NSP', final_number)

    reviews = [val for val in dataframe["Plus"] if len(str(val)) > 4]
    for i, val in dataframe["Plus"].items():
        if len(str(val)) < 4:
            random_review = rd.choice(reviews)
            dataframe.at[i, 'Plus'] = random_review

def save_modified_data(dataframe, file_path):
    dataframe.to_excel(file_path, index=False)
    print(f"Modified dataset saved to {file_path}")

def main():

    dataframe1 = pd.read_excel(r".\\Data cat personality and predation Cordonnier et al.xlsx")

    to_number('Sexe', dataframe1)
    to_number('Nombre', dataframe1)
    to_number('Age', dataframe1)
    to_number('Zone', dataframe1)
    to_number('Logement', dataframe1)

    handle_missing_values(dataframe1)

    modified_file_path = r".\\Modified_Data_cat_personality.xlsx"
    save_modified_data(dataframe1, modified_file_path)

if __name__ == "__main__":
    main()