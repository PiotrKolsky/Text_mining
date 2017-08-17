

################### STAGE II - Data preparing ############################
#########################################################################

import pandas as pd
from datetime import datetime
from datetime import date
import os
import nltk
import re
import string
import pickle
os.chdir('/home/sas/Zasoby/Python/1Dyplom')

### Explained labels, set week number as the index
wig = pd.read_csv('wig_w.csv')
wig = wig.drop(['Otwarcie','Najwyzszy','Najnizszy','Wolumen'], axis=1)
wig['Change'] = 0.0
wig['Gain'] = 0
for i in range(1,len(wig)):
    wig['Change'][i] = wig['Zamkniecie'][i]/wig['Zamkniecie'][i-1]
for i in range(1,len(wig)):
    if wig['Change'][i]>1:
         wig['Gain'][i] = 1
wig = wig.tail(52)
wig.iloc[:,0] = wig.iloc[:,0].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

wig = wig.reset_index(drop=True)
wig.to_csv('Wig.csv', index=False) #backup

### Explanatory variables - date errors fixing and timestamp conversion
forsal = pd.read_csv('Forsal2107.csv')

### Date fixing - converting months names into numbers
for i in range(1,len(forsal)):
    forsal.iloc[i,0] = re.sub(r'(\w+ \d{,4})', forsal.iloc[i-1,0], forsal.iloc[i,0])

### Date string into datetime conversion
def date_conversion(x):
    try:
        x = datetime.strptime(x.strip(), '%d.%m.%Y')
    except:
        x = datetime.strptime(x.strip(), '%Y-%m-%d')
    return x

forsal.iloc[:,0] = forsal.iloc[:,0].map(lambda x: date_conversion(x))

### Cutting date outside analysed period and doubled articles removal
forsal = forsal[forsal['1Date'] > datetime.strptime('21.07.2016', '%d.%m.%Y')]
forsal = forsal.drop_duplicates()

### Week number setting
forsal['WeekNo'] = forsal['1Date']
forsal['WeekNo'] = forsal['WeekNo'].map(lambda x: date.isocalendar(x)[1])
forsal.to_csv('ForsalWeekNo.csv', index=False) #backup

### Week number conversion into ordinal numbers
def weeks(x):
    x = x-30
    if x<0:
        x=x+52
    else: x
    return x

forsal['WeekNo'] = forsal['WeekNo'].map(lambda x: weeks(x))

### Articles grouping by week number
group = forsal.iloc[:].groupby(['WeekNo'])['3Text'].sum()
groupDf = pd.DataFrame(group)
groupDf.to_csv('ForsalGroup.csv', index=True) #backup


######### STAGE III - text cleaning #########################
#############################################################
groupDf = pd.read_csv('ForsalGroup.csv')

articles = []
for i in range(len(groupDf)):
    articles.append(groupDf['3Text'][i])

articles = [re.sub(r'<.+?>',' ', str(a)) for a in articles]
articles = [re.sub(r'\{[~\{^\{]+?\}',' ', str(a)) for a in articles]
articles = [re.sub(r'\n',' ', str(a)) for a in articles]
articles = [re.sub(r'\\',' ', str(a)) for a in articles]
articles = [re.sub(r'[\,\.\"\-\']',' ', str(a)) for a in articles]
articles = [re.sub(r'\d',' ', str(a)) for a in articles]
articles = [re.sub(r'\s{2,}',' ', str(a)) for a in articles]
articles = [nltk.word_tokenize(a) for a in articles] # word detecting
articles = [[w for w in a if w not in string.punctuation] for a in articles]
articles = [[w.lower() for w in a] for a in articles ]
articles = [[w for w in a if len(w)>3] for a in articles]

### Stopwords removal
stopwords =  pd.read_csv('polishStopwords.txt')
articles = [[w for w in a if w not in stopwords.values] for a in articles ]

### Pickling for backup
with open('articles.txt', 'wb') as ap:
    pickle.dump(articles, ap)
#with open('articles.txt', 'rb') as ap:   # Unpickling
#    articles = pickle.load(ap)


### Stemming == change into primary forms in Polish
### Dictionary preparation
words = pd.read_csv('plWordsUTF.txt', sep='\t')
words = words.merge(words, left_on='forma_podst', right_on='id', how='inner') \
    .drop(['id_x','forma_podst_x','id_y','forma_podst_y'], axis=1)
words.columns = ['indexes', 'word']
words = words.set_index(['indexes'])
dicty = words.to_dict(orient='dict')

### Change into primary forms
def changer(wrd):
    if wrd in dicty['word']:
        output = dicty['word'][wrd]
    else: output = wrd
    return output

articles2 = [[changer(w) for w in a] for a in articles]

with open('articles.txt', 'wb') as ap:   #Pickling - backup
    pickle.dump(articles2, ap)

############################################################
