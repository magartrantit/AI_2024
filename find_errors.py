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

if __name__ == "__main__":
    main()
