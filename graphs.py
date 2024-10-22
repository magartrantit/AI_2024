import pandas as pd
import matplotlib.pyplot as plt


def main():

    dataframe1 = pd.read_excel(r"./Modified_Data_cat_personality.xlsx")

    print("Coloane in dataset:", dataframe1.columns)
    
    grouping_column = 'Race'
    numeric_columns = ['Sexe', 'Age', 'Nombre','Logement','Zone','Ext', 'Obs', 'Timide', 'Calme','Effrayé','Intelligent','Vigilant','Perséverant','Affectueux','Amical','Solitaire', 'Brutal','Dominant', 'Agressif','Impulsif','Prévisible','Distrait','Abondance','PredOiseau','PredMamm']

    # Dam drop la randurile care contin valori lipsa
    cleaned_df = dataframe1[[grouping_column] + numeric_columns].dropna()
    


    # STACKED HISTOGRAM

    cleaned_df.groupby(grouping_column)[numeric_columns].sum().T.plot(
        kind='bar', stacked=True, edgecolor='black', figsize=(14, 7)
    )
    
    plt.title('Stacked Histogram of Personality Traits by Race')
    plt.xlabel('Personality Traits')
    plt.ylabel('Sum of Scores')
    plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()



    #BOXPLOT

    num_plots = len(numeric_columns)
    
    cols_per_row = 5
    rows = (num_plots // cols_per_row) + (num_plots % cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        cleaned_df.boxplot(column=column, by=grouping_column, vert=False, ax=axes[i])
        axes[i].set_title(column, fontsize=8)
        #axes[i].set_xlabel('Trait Scores', fontsize=6)
        axes[i].set_ylabel('Race', fontsize=0)
        axes[i].tick_params(axis='x', labelsize=6)
        axes[i].tick_params(axis='y', labelsize=6)

    
    
    # Add a global title and adjust layout
    fig.suptitle('Boxplot of Personality Traits by Race', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()
if __name__ == "__main__":
    main()
