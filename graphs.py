import pandas as pd
import matplotlib.pyplot as plt

# COLOANELE DUPA CE TRANSFORMAM TOTUL IN VALORI NUMERICE
    # ['Sexe', 'Age', 'Nombre','Longement','Zone','Ext', 'Obs', 'Timide', 'Calme','Effrayé','Intelligent','Vigilant','Perséverant','Affectueux','Amical','Solitaire', 'Brutal','Dominant', 'Agressif','Impulsif','Prévisible','Distrait','Abondance','PredOiseau','PredMamm']  

def main():
  
    # Load the Excel file (with headers)
    dataframe1 = pd.read_excel(r".\Data cat personality and predation Cordonnier et al.xlsx")
    
    # Print the column names to verify
    print("Columns in the dataset:", dataframe1.columns)
    
    # Define your columns correctly after verifying them
    grouping_column = 'Race'
    numeric_columns = ['Timide', 'Calme', 'Brutal', 'Agressif']

    # Drop rows with missing values in the relevant columns
    cleaned_df = dataframe1[[grouping_column] + numeric_columns].dropna()
    
    # Create the stacked histogram and set the size of the figure
    cleaned_df.groupby(grouping_column)[numeric_columns].sum().T.plot(
        kind='bar', stacked=True, edgecolor='black'
    )
    
    plt.title('Stacked Histogram of Personality Traits by Race')
    plt.xlabel('Personality Traits')
    plt.ylabel('Sum of Scores')
    plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show the stacked histogram
    plt.show()

    #BOXPLOT

    # Boxplot for numeric columns grouped by 'Race' with separate subplots for each numeric column
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=False, sharey=False)  

    # Boxplot for 'Timide'
    cleaned_df.boxplot(column='Timide', by=grouping_column, vert=False, ax=axes[0, 0])
    axes[0, 0].set_title('Timide')
    axes[0, 0].set_xlabel('Trait Scores')
    axes[0, 0].set_ylabel('Race')

    # Boxplot for 'Calme'
    cleaned_df.boxplot(column='Calme', by=grouping_column, vert=False, ax=axes[0, 1])
    axes[0, 1].set_title('Calme')
    axes[0, 1].set_xlabel('Trait Scores')
    axes[0, 1].set_ylabel('Race')

    # Boxplot for 'Brutal'
    cleaned_df.boxplot(column='Brutal', by=grouping_column, vert=False, ax=axes[1, 0])
    axes[1, 0].set_title('Brutal')
    axes[1, 0].set_xlabel('Trait Scores')
    axes[1, 0].set_ylabel('Race')

    # Boxplot for 'Agressif'
    cleaned_df.boxplot(column='Agressif', by=grouping_column, vert=False, ax=axes[1, 1])
    axes[1, 1].set_title('Agressif')
    axes[1, 1].set_xlabel('Trait Scores')
    axes[1, 1].set_ylabel('Race')

    # Overall title and layout adjustments
    fig.suptitle('Boxplot of Personality Traits by Race', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
