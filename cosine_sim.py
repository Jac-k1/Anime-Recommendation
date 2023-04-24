import os
import numpy as np
import pandas as pd
import warnings
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

pd.options.display.max_columns

rating_df = pd.read_csv('./rating2.csv')
anime_df = pd.read_csv('./anime2.csv')

print(anime_df.head())
print('*'*50)
print(rating_df.head())


print("\n","*"*50)
print("anime_df missing values(%) :\n")
print(round(anime_df.isnull().sum().sort_values(ascending=False)/len(anime_df.index),4)*100) 
print("\n","*"*50,"\n\nrating_df missing values (%):\n")
print(round(rating_df.isnull().sum().sort_values(ascending=False)/len(rating_df.index),4)*100)
print("\n","*"*50)
#print(anime_df['type'].mode())
#print(anime_df['genre'].mode())

# deleting anime with 0 rating
anime_df=anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type5
anime_df['genre'] = anime_df['genre'].fillna(
anime_df['genre'].dropna().mode().values[0])

anime_df['type'] = anime_df['type'].fillna(
anime_df['type'].dropna().mode().values[0])

#checking if all null values are filled
anime_df.isnull().sum()

rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x==-1 else x)
print(rating_df.head(10))

#**************************************************************************************************************************************************************************

# Making DataFrame for recommendation
#### recommend only relavent type, in this case we recommend from TV series (maybe, maybe not used)
# make the new DataFrame with only the features we need
# Use # users for runtime 

#step 1
#anime_df = anime_df[anime_df['type']=='TV']  # this recommends only one type of anime, i decided to include all types and not just TV
#step 2
rated_anime = rating_df.merge(anime_df, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
#step 3
rated_anime =rated_anime[['user_id', 'name', 'rating']]
#step 4
rated_anime_num= rated_anime[rated_anime.user_id <= 7000] # we can change this value
rated_anime_num.head()

#**************************************************************************************************************************************************************************

# Pivot table helps us get a visual table with users as rows and showname as columns. We can see what shows they have rated and shows they havent
pivot = rated_anime_num.pivot_table(index=['user_id'], columns=['name'], values='rating')
print('pivot table snippet')
print(pivot.head())

#**************************************************************************************************************************************************************************

# We need to work on our Pivot Table
# normalize the values
# instead of Nan, we convert them to zeros
# Then we drop the columns with zeros for shows they havent rated
# We can convert to sparse matrix for similarity comparison (since we have a lot of zeros)

# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
# step 2
pivot_n.fillna(0, inplace=True)
# step 3
pivot_n = pivot_n.T
# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]
# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)

#model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)
print('anime Similarity')
print(anime_similarity)


#Df of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)

#**************************************************************************************************************************************************************************

'''
We make a function that can return the top n shows with highest cosine similarity value and the percentage of similarity.
    Input: would be the anime name

    Output: list of anime similar and their similarity percentage
'''

def recommend(ani_name):
    number = 1
    print('Recommendations for {}:\n'.format(ani_name))
    for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:21]:
        print(f'#{number}: {anime} - {round(ani_sim_df[anime][ani_name]*100,2)}% match')
        number +=1  
        
    print("*"*50)


recommend('Sword Art Online')

recommend('Shingeki no Kyojin')

recommend('Death Note')