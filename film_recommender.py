#                                   Content-Based Recommender using NLP
# A step-by-step approach to creating a content-based recommender system using natural language processing (NLP).
# The dataset is taken from data.world: https://data.world/studentoflife/imdb-top-250-lists-and-5000-or-so-data-records ,
# and from kaggle datasets "download -d nareshbhat/indian-moviesimdb" which is scraped from IMDB.
# Title is the attributes used to compare similarity.

# Step 1: Import Python libraries and dataset and perform Exploratory Data Analysis(EDA)
"""
For this project we have to make sure that we have installed RAKE (Rapid Automatic Keyword Extraction)
library or in short "pip install rake_nltk".

RAKE Algorithm:

RAKE is a domain-independent keyword extraction technique that analyses the frequency of word appearance and
its co-occurrence with other terms in the text to try to discover important phrases in a body of text.
"""

from rake_nltk import Rake
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('IMDB_TopIndianmovies2_OMDB_Detailed.csv')
print(df)

# data overview
print('Rows x Columns : ', df.shape[0], 'x', df.shape[1])
print('Features: ', df.columns.tolist())
print('\nUnique Value:')
print(df.nunique())
for col in df.columns:
    print(col, end=': ')
    print(df[col].unique())

# What sort of entries are there, and how many missing values/null fields are there?
df.info()
print('\nValues Missing:  ', df.isnull().sum().values.sum())
df.isnull().sum()

# for all number columns, a summary statistic
print(df.describe().T)

# Keep only these 5 relevant columns, together with 250 rows without a NaN field.
df = df[['Title', 'Year', 'Genre', 'Language']]
print(df)

# Selecting Genre from the Data Set:
print(df.loc[(df.Genre == 'Action')])

# the most popular genres (from 110 unique genres)
print(df['Genre'].value_counts())

"""
# Step 2: Pre-Processing Data
  ---------------------------
To begin, the data must be pre-processed using natural language processing (NLP) to produce only one column containing 
all of the movie's properties (in words). After that, vectorization is used to transform the data into numbers, 
and each word is given a score. The cosine similarity may then be computed.
"""

# to remove white space, punctuation,stop words and convert all words to lower case (Applying RAKE Algorithm)¶

"""
In the Plot column, Rake function is used to extract the most relevant words from full sentences. 
To do so, I used this function on each row in the Plot column and put the list of key words in a new column called 'Key words'.
"""

# To eliminate punctuations from Language
df['Language'] = df['Language'].str.replace('[^\w\s]', '')


# to convert Title into a list of essential words
df['KeyWords'] = ''  # setting up a new column
r = Rake()  # Rake can be used to get rid of stop words (based on english stopwords from NLTK)
for index, row in df.iterrows():
    r.extract_keywords_from_text(row['Title'])  # To extract essential keywords from Title, use lower case by default.
    keyWordsDictScores = r.get_word_degrees()  # to get a dictionary that includes key words and their scores
    row['KeyWords'] = list(keyWordsDictScores.keys())  # to add a new column with a list of key words
print(df)

print()

# to view the last item in the list:
print("Last Movie in the List:")
print(df['Title'][2249])

print()
# to see the most recent dictionary from Plot
print(keyWordsDictScores)

print()

# to display the most recent item in KeyWords
print(df['KeyWords'][249])
print()


# to create a list of unique Titles, to eliminate duplicates and finally to transform the word into lowercase:
df['Title'] = df['Title'].map(lambda x: x.split(','))
for index, row in df.iterrows():
    row['Title'] = [x.lower().replace(' ', '') for x in row['Title']]
print(df)
print()

"""
# Step 3: combine column attributes to Bag_of_words to produce a word representation
  ----------------------------------------------------------------------------------
"""
# to merge four lists of key terms (four columns) into one statement in the Bag_of_words column
df['Bag_of_words'] = ''
columns = ['Genre', 'Title', 'Language', 'KeyWords']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words

# Whitespaces in front and behind are removed, and numerous whitespaces are replaced (if any)
df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

df = df[['Title', 'Bag_of_words']]
print(df)

print()

# an example of what may be found in the Bag_of_words
print(df['Bag_of_words'][0])

print()
"""
Step 4: build a vector representation of Bag_of_words as well as a similarity matrix
------------------------------------------------------------------------------------
We need to transform the 'Bag of words' into vector representation using CountVectorizer,
which is a basic frequency counter for each word in the bag of words column, 
because the recommender model can only read and compare vectors (matrices). 
I can use the cosine similarity function to examine similarities between movies once I get the matrix with the count for each word.
"""
# to create a count matrix:
sum = CountVectorizer()
matrixCount = sum.fit_transform(df['Bag_of_words'])
print(matrixCount)
print()

# to create a matrix of cosine similarity (size 250 x 250)
# All movies are represented by rows, while all movies are represented by columns.
# similarity in cosine: similarity = cos(angle) = 0 (difference) to 1 (similarity) (similar)
# Because every movie is identical to itself, all the numbers on the diagonal are 1. (cosine value is 1 means exactly identical)
# The similarity between A and B is the same as the similarity between B and A, hence the matrix is symmetrical.
# Movies x and y have a similarity value of 0.1578947 for different values, such as 0.1578947.

cosine_similarity = cosine_similarity(matrixCount, matrixCount)
print(cosine_similarity)

print()

"""
Similarity Matrix:
-----------------
The next step is to make a Series of movie titles, with the series index matching the similarity matrix's row/column index.
"""
# to make a movie title series that may be used as indexes (each index is mapped to a movie title)
indices = pd.Series(df['Title'])
print(indices[:7])

"""
Step 5: run and test the recommender model
------------------------------------------
"""
print()

# This method accepts a film title and returns the top 13 recommended (similar) films.
def suggest(title, cosine_similarity=cosine_similarity):
    recommended_films = []
    indxx = indices[indices == title].index[0]  # to find the movie title index that matches the provided movie
    score_series = pd.Series(cosine_similarity[indxx]).sort_values(ascending=False)  # Similarity ratings are shown in order of decreasing similarity.
    top_13_indices = list(score_series.iloc[1:12].index)  # to obtain the indices of the top ten most identical films
    # [1:14] to exclude 0 (index 0 is the input movie itself)

    for i in top_13_indices:  # to append the titles of top 13 similar movies to the recommended_films list
        recommended_films.append(list(df['Title'])[i])

    return recommended_films

print(suggest('Anand'))


"""
*** Reference ***
Influenced by the work of James Ng,
https://github.com/JNYH/movie_recommender
"""