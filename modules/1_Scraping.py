
### STAGE I - web scrapping and parsing ############
####################################################

from bs4 import BeautifulSoup
import requests, os
import pandas as pd
import re
os.chdir('/home/sas/Zasoby/Python/1Dyplom')

### Table structure seting
df = pd.DataFrame(columns=['1Date','2Title','3Text'])

### Links searching
for j in range(1,10):
    url = 'http://forsal.pl/wydarzenia,'+str(j)
    print(url)
    f = requests.get(url)
    soup = BeautifulSoup(f.text, 'html.parser')
    mat = str(soup('h3', class_='open')[:])
    link = (re.findall(r"<a.*? href=\"(.*?)\".*?>(.*?)</a>", mat)[:])
    ### Consecutive article loading
    for i in range(len(link)):
        if 'swiat' not in link[i][0] and 'galerie' not in link[i][0] and 'tabela' not in link[i][0]:
            print(link[i][0])
            url2 = link[i][0]
            f = requests.get(url2)
            soup = BeautifulSoup(f.text, 'html.parser')

            tytul = soup('h1', class_='open title', itemprop='headline')[0].text
            data = soup.find_all(class_='date')[0].text[-17:-7]
            try:
                tekst = soup.find_all(class_='leadDiv')[0].text[:]+' '+soup.find_all(class_='articleBody')[0].text.split('>>>')[0]
            except:
                tekst = soup.find_all(class_='lead')[0].text[:]+' '+soup.find_all(class_='articleBody')[0].text.split('>>>')[0]

            df2 = pd.DataFrame({'1Date':[data], '2Title':tytul, '3Text':tekst})
            df = df.append(df2)

### Writing down the table
df = df.reset_index(drop=True)
df.to_csv('Forsal2107.csv', index=False)

#with open('Forsal2107.csv', 'a') as f:
#    df.to_csv(f, header=False, index=False)

#df2 = pd.read_csv('Forsal2107.csv')
####################################
