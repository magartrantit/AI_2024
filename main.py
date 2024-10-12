import pandas as pd



def main():
    
    dataframe1 = pd.read_excel(".\Data cat personality and predation Cordonnier et al.xlsx")

    # print(dataframe1)
    
    for column in dataframe1.columns:
        missing_data = dataframe1[column].isnull()
        
        if missing_data.any():
            print(f"Coloana '{column}' are date lipsă la următoarele rânduri:")
            print(dataframe1[missing_data].index.tolist())
        else:
            print(f"Coloana '{column}' nu are date lipsă.")
    
    

if __name__ == "__main__":
    main()
