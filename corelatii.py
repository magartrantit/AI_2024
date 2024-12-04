import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel(".\\Modified_Data_cat_personality.xlsx")

data = data.iloc[:, 2:-1]

numeric_data = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_data.corr().round(2)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap of Correlation Matrix')
plt.show()
