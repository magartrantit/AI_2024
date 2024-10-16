import pandas as pd
from sklearn.preprocessing import LabelEncoder
def to_number(attribute, dataframe):
    enc = LabelEncoder()
    enc.fit(dataframe[attribute])
    dataframe[attribute] = enc.transform(dataframe[attribute])

def main():
    # Load the Excel file
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")

    #to_number('Race', dataframe1)
    to_number('Sexe', dataframe1)
    to_number('Nombre', dataframe1)
    to_number('Age', dataframe1)
    to_number('Zone', dataframe1)
    print(dataframe1)

if __name__ == "__main__":
    main()
