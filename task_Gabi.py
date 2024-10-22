import pandas as pd
import random as rd



def main():
    
    dataframe1 = pd.read_excel(".\Data cat personality and predation Cordonnier et al.xlsx")

    sum = 0
    count = 0
    
    reviews = []
    count_reviews = 0
        
    for column in dataframe1.columns:  # Pasul 1: calculăm suma valorilor si numărul lor
        if column == "Abondance":
            print("Suntem aici")
            for val in dataframe1[column]:
               if(val != 'NSP'):
                  sum = sum +  int(val)
                  count += 1
        
    # print("Suma este: ", sum , " Iar count ul este: ", count)
    final_number = round(sum / count,2)
    # print(final_number)
    
    dataframe1["Abondance"] = dataframe1["Abondance"].replace("NSP", final_number) # Pasul 2: înlocuim NSP cu numărul dat de formulă
    # print(dataframe1)
    
    
    #Problema 2: Adăugăm review-uri aleatorii
    
    for column in dataframe1.columns:
        if(column == "Plus"):
            for val in dataframe1[column]:
                if(len(str(val)) > 4):
                    count_reviews += 1
                    reviews.append(val)
    
    # print(count_reviews, " ", reviews[810])
    
    for column in dataframe1.columns:
        if(column ==  "Plus"):
           for i, val in dataframe1[column].items():
               if(len(str(val)) < 4):
                   random_number = rd.randint(0, count_reviews - 1)
                   dataframe1.loc[i, column] = reviews[random_number]
    
    
    print(dataframe1)
                
                    
                    
                    
                    
if __name__ == "__main__":
    main()