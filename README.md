# Text Mining Project

1. Purposes of the project

The aim of the project was to test the possibility of forecasting future change of Warsaw Stock Exchange Index (WIG), depending on arcticles content. The term frequency of particular words in articles was used as a explanatory variables.
Rather than exact percent of index growth, it was decided to classify respective week press content to day index change classification [1/0], where:
- 1 means daily index growth (buy signal), the profit at the end of day is expected,
- 0 means index falling (sell signal), the loss at the end of day is forecasted.

Survey performed on July/September 2017 using previous 246 days data, on Python 3.5.

2. Data description

Labels for classification:
- Warsaw Stock Exchange Index WIG (daily closing index value) from 29.07.2016 to 21.07.2017. 
- category label 'gain' has been coded:
	- index rise = buy signal = 1,
	- index fall or no change = sell signal = 0.

Explanatory data:
- 34000 articles of 22.07.2016 - 21.07.2017 period have been downloaded;
- Then were grouped by 1, 2 up to 7 days before index date, stemmed and stopwords removed. 

3. Scope of work

Stage I. Articles loading 
- extracting links from the home page of the Forsal economic web portal;
- loading consecutive articles;
- articles text extracted using BeautifulSoup package.

Stage II - Data cleaning
- calculating labels 1/0 for WIG index data;
- fixing dates of the articles and converting into datetime;
- grouping into 1-7 day periods.

Stage III - Text preparing for vectorisation
- removing of white and functional signs;
- stemming - changing particular words into their basic form in Polish;
- stopwords removal.

Stage IV - Machine learning
- word chains vectorisation using countvectorizer;
- dimensionality reduction using truncatedSVD
- preparing the classificators list;
- calculating the accuracy performance measure using cross validation:
	- kFold,
	- leaveOneOut,
- parameters tuning by GridSearch for choosen classifiers;
- ROC curves drawing and AUC calculation;
- final parameters tuning by GridSearch and Pipeline.

Forecasting
- some word chains were added as a testing set:
	- with positive investment tenor (label = 1 expected),
	- with negative and neutral investment tenor (label = 0 expected),
- labels were predicted.

4. Results and conclusions
- 1 previous day articles optimal as the explanatory variables
- dimensionality reduction to 6 first dimensions (of total 148k)

- preliminary average accuracy scores:

Name  mean_accuracy  (std_accuracy)

LR       0.573  		(0.0656)

KNN      0.561  		(0.0638)

CART     0.533  		(0.0502)

RFC      0.508  		(0.0716)

ETC      0.569  		(0.0709)

AdaBoost 0.480 			(0.0287)

SVMlin   0.561  		(0.0623)

SVMrbf   0.557  		(0.0724)

NNlog    0.569  		(0.0510)

NNrelu   0.569  		(0.0519)

- average accuracy scores for different params:

Model:  LR

Rank: 1; params: 'C': 1; mean accuracy: 0.61; std accuracy: 0.02

Rank: 2; params: 'C': 0.1; mean accuracy: 0.60; std accuracy: 0.02

Rank: 2; params: 'C': 10; mean accuracy: 0.60; std accuracy: 0.02

Model:  CART

Rank: 1; params: 'max_depth': None; mean accuracy: 0.53; std accuracy: 0.05

Rank: 2; params: 'max_depth': 6; mean accuracy: 0.53; std accuracy: 0.03

Rank: 3; params: 'max_depth': 3; mean accuracy: 0.51; std accuracy: 0.03

Rank: 4; params: 'max_depth': 10; mean accuracy: 0.49; std accuracy: 0.03

Model:  RFC

Rank: 1; params: 'n_estimators': 30; mean accuracy: 0.57; std accuracy: 0.04

Rank: 2; params: 'n_estimators': 50; mean accuracy: 0.52; std accuracy: 0.02

Rank: 3; params: 'n_estimators': 70; mean accuracy: 0.50; std accuracy: 0.04

Model:  KNN

Rank: 1; params: 'n_neighbors': 3; mean accuracy: 0.52; std accuracy: 0.04

Rank: 2; params: 'n_neighbors': 2; mean accuracy: 0.50; std accuracy: 0.02

Rank: 3; params: 'n_neighbors': 5; mean accuracy: 0.50; std accuracy: 0.05

Model:  SVC

Rank: 1; params: 'kernel': 'rbf'; mean accuracy: 0.56; std accuracy: 0.00

Rank: 2; params: 'kernel': 'linear'; mean accuracy: 0.56; std accuracy: 0.02

- finally linear regression model has been choosen by pipeline, with average accuracy 0.602 (C=1, countvectorizer ngram_range=(1,1), truncatedSVD n_components=6).

5. Appendix - search engine & recommender
- to facilitate content management and read articles on certain subject area, simple search engine and recomendation engine was added, both based on cosine similarity.
