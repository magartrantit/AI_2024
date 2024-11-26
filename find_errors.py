import pandas as pd
def correspondance(file_path):

    data_df = pd.read_excel(file_path, sheet_name='Data')
    code_df = pd.read_excel(file_path, sheet_name='Code')
    data_df.head(), code_df.head()

    code_with_options = code_df.dropna(subset=['Values']).copy()  
    code_with_options['Values'] = code_with_options['Values'].str.split('/')

    columns_to_check = code_with_options[['Variable', 'Values']].set_index('Variable')['Values'].to_dict()

    mismatch_dict = {}

    for col, valid_values in columns_to_check.items():
        if col in data_df.columns:
            unique_data_values = data_df[col].dropna().unique()
            mismatched_values = [val for val in unique_data_values if str(val) not in valid_values]
            if mismatched_values:
                mismatch_dict[col] = mismatched_values

    mismatch_df = pd.DataFrame(list(mismatch_dict.items()), columns=['Column', 'Mismatched Values'])
    print(mismatch_df)

def main():

    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")
    for column in dataframe1.columns:
        missing_data = dataframe1[column].isnull()
        
        if missing_data.any():
            print(f"Coloana '{column}' are date lipsă la următoarele rânduri:")
            print(dataframe1[missing_data].index.tolist())
    

    cat_breeds_column = dataframe1.iloc[1:, 4] 

    cat_breed_counts = cat_breeds_column.value_counts()

    print("\nNumărul de instanțe pentru fiecare rasă de pisici (prescurtat):")
    print(cat_breed_counts)

    dataframe_filtered = dataframe1.iloc[:, 2:-1]


    print("Valori distincte și frecvențe pentru fiecare atribut:")
    for column in dataframe_filtered.columns:
        distinct_values = dataframe_filtered[column].value_counts()
        print(f"\nAtributul '{column}':")
        print(f"Număr total de valori distincte: {len(distinct_values)}")
        print("Frecvența fiecărei valori:")
        print(distinct_values)


    print("\nValori distincte și frecvențe la nivel de rasă de pisici:")
    for breed in cat_breed_counts.index:
        breed_data = dataframe1[dataframe1['Race'] == breed].iloc[:, 2:-1]
        print(f"\nPentru rasa de pisici '{breed}':")
        for column in breed_data.columns:
            distinct_values = breed_data[column].value_counts()
            print(f"Atributul '{column}':")
            print(f"Număr total de valori distincte: {len(distinct_values)}")
            print("Frecvența fiecărei valori:")
            print(distinct_values)

    columns_to_ignore = ['Row.names', 'Horodateur', 'Plus']
    df_filtered = dataframe1.drop(columns=columns_to_ignore)

    duplicates = df_filtered[df_filtered.duplicated(keep=False)]


    if not duplicates.empty:
        print("\nInstanțe duplicate găsite:")
        print(duplicates)
        duplicates.to_csv('duplicates.csv', index=False)
    else:
        print("\nNu au fost găsite instanțe duplicate.")


    correspondance(r".\Data cat personality and predation Cordonnier et al.xlsx")


if __name__ == "__main__":
    main()