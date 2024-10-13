import pandas as pd

def main():
    # Load the Excel file
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx", header=None)
    
    # 1. Find missing data in each column
    for column in dataframe1.columns:
        missing_data = dataframe1[column].isnull()
        
        if missing_data.any():
            print(f"Coloana '{column}' are date lipsă la următoarele rânduri:")
            print(dataframe1[missing_data].index.tolist())
    
    # 2. Count instances of each cat breed
    # Assuming the breeds are in the fifth column (index 4 since Python is 0-indexed)
    cat_breeds_column = dataframe1.iloc[1:, 4]  # Select column by index
    
    # Count occurrences of each breed
    cat_breed_counts = cat_breeds_column.value_counts()
    
    print("\nNumărul de instanțe pentru fiecare rasă de pisici (prescurtat):")
    print(cat_breed_counts)

    dataframe_filtered = dataframe1.iloc[:, 2:-1]

    # 3. Afișarea valorilor distincte și frecvența acestora pentru fiecare atribut
    print("Valori distincte și frecvențe pentru fiecare atribut:")
    for column in dataframe_filtered.columns:
        distinct_values = dataframe_filtered[column].value_counts()
        print(f"\nAtributul '{column}':")
        print(f"Număr total de valori distincte: {len(distinct_values)}")
        print("Frecvența fiecărei valori:")
        print(distinct_values)

    # 4. Afișarea valorilor distincte și frecvențelor pentru fiecare rasă de pisici
    print("\nValori distincte și frecvențe la nivel de rasă de pisici:")
    for breed in cat_breed_counts.index:
        breed_data = dataframe1[dataframe1[4] == breed].iloc[:, 2:-1]
        print(f"\nPentru rasa de pisici '{breed}':")
        for column in breed_data.columns:
            distinct_values = breed_data[column].value_counts()
            print(f"Atributul '{column}':")
            print(f"Număr total de valori distincte: {len(distinct_values)}")
            print("Frecvența fiecărei valori:")
            print(distinct_values)



if __name__ == "__main__":
    main()
