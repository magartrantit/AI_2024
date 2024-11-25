import pandas as pd
import matplotlib.pyplot as plt


def main():

    dataframe1 = pd.read_excel(r"./Modified_Data_cat_personality.xlsx")

    print("Coloane in dataset:", dataframe1.columns)
    
    grouping_column = 'Race'
    numeric_columns = ['Sexe', 'Age', 'Nombre','Logement','Zone','Ext', 'Obs', 'Timide', 'Calme','Effrayé','Intelligent','Vigilant','Perséverant','Affectueux','Amical','Solitaire', 'Brutal','Dominant', 'Agressif','Impulsif','Prévisible','Distrait','Abondance','PredOiseau','PredMamm']

    cleaned_df = dataframe1[[grouping_column] + numeric_columns].dropna()

    cleaned_df.groupby(grouping_column)[numeric_columns].sum().T.plot(
        kind='bar', stacked=True, edgecolor='black', figsize=(14, 7)
    )
    
    plt.title('Stacked Histogram of Personality Traits by Race')
    plt.xlabel('Personality Traits')
    plt.ylabel('Sum of Scores')
    plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    half = len(numeric_columns) // 2
    first_half_columns = numeric_columns[:half]
    second_half_columns = numeric_columns[half:]

    num_plots = len(first_half_columns)
    cols_per_row = 4
    rows = (num_plots + cols_per_row - 1) // cols_per_row
    fig1, axes1 = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3))
    axes1 = axes1.flatten()

    for i, column in enumerate(first_half_columns):
        cleaned_df.boxplot(column=column, by=grouping_column, vert=False, ax=axes1[i])
        axes1[i].set_title(column, fontsize=8)
        axes1[i].set_ylabel('Race', fontsize=0)
        axes1[i].tick_params(axis='x', labelsize=6)
        axes1[i].tick_params(axis='y', labelsize=6)

    for j in range(i + 1, len(axes1)):
        fig1.delaxes(axes1[j])

    fig1.suptitle('Boxplot of Personality Traits by Race (Set 1)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    num_plots = len(second_half_columns)
    rows = (num_plots + cols_per_row - 1) // cols_per_row  
    fig2, axes2 = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3))
    axes2 = axes2.flatten()

    for i, column in enumerate(second_half_columns):
        cleaned_df.boxplot(column=column, by=grouping_column, vert=False, ax=axes2[i])
        axes2[i].set_title(column, fontsize=8)
        axes2[i].set_ylabel('Race', fontsize=0)
        axes2[i].tick_params(axis='x', labelsize=6)
        axes2[i].tick_params(axis='y', labelsize=6)

    for j in range(i + 1, len(axes2)):
        fig2.delaxes(axes2[j])

    fig2.suptitle('Boxplot of Personality Traits by Race (Set 2)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == "__main__":
    main()
