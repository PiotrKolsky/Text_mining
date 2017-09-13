
################### STAGE II - Data preparing ############################
#########################################################################

import pandas as pd
from datetime import datetime, timedelta
from datetime import date
import os
import nltk
import re
import string
import pickle
os.chdir('/home/sas/Zasoby/Python/1Dyplom')

### Explained labels, data labels [0/1] preparation (stock exchange index changes)
wig = pd.read_csv('wig_d.csv') # Raw data
wig = wig.drop(['Otwarcie','Najwyzszy','Najnizszy','Wolumen'], axis=1)
wig['Change'] = 0.0
wig['Gain'] = 0
for i in range(1,len(wig)):
    wig['Change'][i] = wig['Zamkniecie'][i]/wig['Zamkniecie'][i-1]
for i in range(1,len(wig)):
    if wig['Change'][i]>1:
         wig['Gain'][i] = 1

### Date converting & cutting
wig.iloc[:,0] = wig.iloc[:,0].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
wig = wig[wig['Data'] > datetime.strptime('28.07.2016', '%d.%m.%Y')]

wig = wig.reset_index(drop=True)
wig.to_csv('Wig.csv', index=False) #backup

### Explanatory variables (press articles) - date errors fixing and timestamp conversion
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

### Backup commands
forsal.to_csv('ForsalDates.csv', index=False) #backup
#wig = pd.read_csv('Wig.csv')
#forsal = pd.read_csv('ForsalDates.csv')
#wig.iloc[:,0] = wig.iloc[:,0].map(lambda x: datetime.strptime(x.strip(), '%Y-%m-%d'))
#forsal.iloc[:,0] = forsal.iloc[:,0].map(lambda x: datetime.strptime(x.strip(), '%Y-%m-%d'))


### Extracting articles in selected period of earlier days moving time window
period = 1
forsalDate = pd.DataFrame(columns=['1Date','2Title','3Text','GroupDate'])
for d in wig.Data:
    print('---', d)
    dfTemp = pd.DataFrame(columns=['1Date','2Title','3Text','GroupDate'])
    for i in range(1,period+1):
        print(d-timedelta(days=i))
        df2 = forsal[forsal['1Date']==(d-timedelta(days=i))]
        df2['GroupDate'] = d
        dfTemp = dfTemp.append(df2)
    forsalDate = forsalDate.append(dfTemp)

### Grouping articles in particular time windows
group = forsalDate.iloc[:].groupby(['GroupDate'])['3Text'].sum()
group = pd.DataFrame(group)
group.to_csv('ForsalGroup1Days.csv', index=True) #another backup
group.head()

######### STAGE III - text cleaning #########################
#############################################################
group = pd.read_csv('ForsalGroup1Days.csv')

articles = []
for i in range(len(group)):
    articles.append(group['3Text'][i])
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

### Stemming == change into primary forms in Polish
### Dictionary preparation
words = pd.read_csv('plWordsUTF.txt', sep='\t')
words = words.merge(words, left_on='forma_podst', right_on='id', how='inner') \
    .drop(['id_x','forma_podst_x','id_y','forma_podst_y'], axis=1)
words.columns = ['indexes', 'word']
words = words.set_index(['indexes'])
dicty = words.to_dict(orient='dict')

### Changing into primary forms
def changer(wrd):
    if wrd in dicty['word']:
        output = dicty['word'][wrd]
    else: output = wrd
    return output

articles = [[changer(w) for w in a] for a in articles]

### Pickling results for machine learning - backup
with open('articles1Days.txt', 'wb') as ap:
    pickle.dump(articles, ap)

############################################################
