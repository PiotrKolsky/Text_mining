
######################## Search engine for content analysis
###########################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import nltk
import re
import string
import pickle
import os
os.chdir('/home/sas/Zasoby/Python/1Dyplom')

forsal = pd.read_csv('ForsalWeekNo.csv')
'''
### Preparing the text of articles
articles = []
for i in range(len(forsal)):
    articles.append(forsal['3Text'][i])

articles = [re.sub(r'<.+?>',' ', str(a)) for a in articles]
articles = [re.sub(r'\{[~\{^\{]+?\}',' ', str(a)) for a in articles]
articles = [re.sub(r'\n',' ', str(a)) for a in articles]
articles = [re.sub(r'\\',' ', str(a)) for a in articles]
articles = [re.sub(r'[\,\.\"\-\']',' ', str(a)) for a in articles]
articles = [re.sub(r'\d',' ', str(a)) for a in articles]
articles = [re.sub(r'\s{2,}',' ', str(a)) for a in articles]
articles = [nltk.word_tokenize(a) for a in articles]
articles = [[w for w in a if w not in string.punctuation] for a in articles]
articles = [[w.lower() for w in a] for a in articles ]
articles = [[w for w in a if len(w)>3] for a in articles]

### Stopwords removal
stopwords =  pd.read_csv('polishStopwords.txt')
articles = [[w for w in a if w not in stopwords.values] for a in articles ]

### Pickling for backup
with open('articlesSearch.txt', 'wb') as ap:
    pickle.dump(articles, ap)
'''
with open('articlesSearch.txt', 'rb') as ap:   # Unpickling
    articles = pickle.load(ap)
articles2 = [' '.join(a) for a in articles]

tfidf = TfidfVectorizer()

### Search engine
def searcher(str, n=10):
    search = str.lower()
    probe = articles2[:]
    probe.append(search)
    search_matrix = tfidf.fit_transform(probe)
    result = cosine_similarity(search_matrix[-1],search_matrix[:-1])

    result = pd.DataFrame(result).T
    result.columns = ['Similar']
    result = result[result.Similar!=0]

    titles = pd.merge(result, forsal.iloc[:,:2], left_index=True, right_index=True).sort_values(by=['Similar', '1Date'], ascending=False);
    print(titles[:n])
    return titles[:n]

### Recommendation engine
def recommender(titles, m=6):
    choosen = forsal.loc[titles.index[0]]['3Text']
    probe = articles2[:]
    probe.append(choosen)
    similar_matrix = tfidf.fit_transform(probe)
    result = cosine_similarity(similar_matrix[-1], similar_matrix[:-1])
    similar = pd.DataFrame(result).T
    similar.columns = ['Similar']
    similar = pd.merge(similar, forsal.iloc[:,:2], left_index=True, right_index=True).sort_values(by=['Similar', '1Date'], ascending=False)
    print(' U may be also interested in reading: \n',similar[1:m+1])
    wcloud(similar[1:m+1])

### Word Cloud for recommendations
def wcloud(simil):
    stext = ' '.join(simil['2Title'])
    wrdcloud = WordCloud(relative_scaling = 1.0).generate(stext)
    plt.figure(figsize=(12,8))
    plt.imshow(wrdcloud)
    plt.axis("off")
    plt.title('Recommendations')
    plt.show()

### Just searching & articles reading
titles = searcher('bankowość') # 'prezydent prawo ustawa podpis' ; 'inwestorzy ucieczka panika'
print(forsal['3Text'].iloc[titles.index[0]]) # Articles reading
recommender(titles)

######################################################
