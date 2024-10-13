import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the Excel file (with headers)
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")
    
    # Filter out non-numeric columns (like 'Row.names', if present)
    numeric_columns = dataframe1.select_dtypes(include='number').columns
    
    # Plot histograms only for numeric columns
    print("Plotting histograms for numeric attributes...")
    dataframe1[numeric_columns].hist(figsize=(20, 20), bins=10, edgecolor='black')
    plt.suptitle('Distribuția valorilor pentru atribute numerice (Histograme)', fontsize=16)
    plt.show()

    # Plot boxplots for numeric columns
    print("Plotting boxplots for numeric attributes...")
    dataframe1[numeric_columns].boxplot(figsize=(20, 10), vert=False)
    plt.title('Boxplot pentru atribute numerice')
    plt.show()

if __name__ == "__main__":
    main()
