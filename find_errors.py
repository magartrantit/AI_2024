import pandas as pd

def main():
    # Încarcă fișierul Excel
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")
    
    # 1. Găsește datele lipsă din fiecare coloană
    for column in dataframe1.columns:
        missing_data = dataframe1[column].isnull()
        
        if missing_data.any():
            print(f"Coloana '{column}' are date lipsă la următoarele rânduri:")
            print(dataframe1[missing_data].index.tolist())
    
    # 2. Numără instanțele pentru fiecare rasă de pisici
    cat_breeds_column = dataframe1.iloc[1:, 4]  # Selectează coloana după index

    cat_breed_counts = cat_breeds_column.value_counts()
    
    print("\nNumărul de instanțe pentru fiecare rasă de pisici (prescurtat):")
    print(cat_breed_counts)

    # Filtrează cadrul de date pentru coloanele relevante
    dataframe_filtered = dataframe1.iloc[:, 2:-1]

    # 3. Afișează valorile distincte și frecvențele pentru fiecare atribut
    print("Valori distincte și frecvențe pentru fiecare atribut:")
    for column in dataframe_filtered.columns:
        distinct_values = dataframe_filtered[column].value_counts()
        print(f"\nAtributul '{column}':")
        print(f"Număr total de valori distincte: {len(distinct_values)}")
        print("Frecvența fiecărei valori:")
        print(distinct_values)

    # 4. Afișează valorile distincte și frecvențele pentru fiecare rasă de pisici
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

    # 5. Elimină coloanele 'Row.names', 'Horodateur', 'Plus' și verifică instanțele duplicate
    columns_to_ignore = ['Row.names', 'Horodateur', 'Plus']
    df_filtered = dataframe1.drop(columns=columns_to_ignore)

    # Verifică duplicatele pe baza coloanelor rămase
    duplicates = df_filtered[df_filtered.duplicated(keep=False)]

    # Afișează instanțele duplicate
    if not duplicates.empty:
        print("\nInstanțe duplicate găsite:")
        print(duplicates)
        # Opțional, salvează duplicatele într-un fișier CSV
        duplicates.to_csv('duplicates.csv', index=False)
    else:
        print("\nNu au fost găsite instanțe duplicate.")


if __name__ == "__main__":
    main()
