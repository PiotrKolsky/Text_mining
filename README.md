# Text Mining Project

1. Purposes of the project

The aim of the project was to test the possibility of forecasting future change of Warsaw Stock Exchange Index (WIG), depending on arcticles content. The term frequency of particular words in articles was used as a explanatory variables.
Rather than exact percent of index growth, it was decided to classify respective week press content to week index change classification [1/0], where:
- 1 means weekly index growth (buy signal), the profit at the end of week is expected,
- 0 means index falling (sell signal), the loss at the end of week is forecasted.

Survey performed on July/September 2017 using previous 52 weeks data, on Python 3.5.

2. Data description

Labels for classification:
- Warsaw Stock Exchange Index WIG (weekly period - closing index value on friday) from 29.07.2016 to 21.07.2017. 
- category label 'gain' has been coded:
	- index rise = buy signal = 1,
	- index fall or no change = sell signal = 0.

Explanatory data:
- 34000 articles of 22.07.2016 - 21.07.2017 period have been downloaded;
- Then were grouped by weeks, stemmed and stopwords removed. 

3. Scope of work

Stage I. Articles loading 
- extracting links from the home page of the Forsal economic web portal;
- loading consecutive articles;
- articles text extracted using BeautifulSoup package.

Stage II - Data cleaning
- calculating labels 1/0 for WIG index data;
- fixing dates of the articles and converting into datetime;
- grouping into weeks.

Stage III - Text preparing for vectorisation
- removing of white and functional signs;
- stemming - changing particular words into their basic form in Polish;
- stopwords removal.

Stage IV - Machine learning
- word chains vectorisation using term frequencies:
	- TFIDF (term frequency – inverse document frequency),
	- TF (term frequency),
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
- for TFIDF vectorisation the best average accuracy scores:
	- LR 0.60
	- KNN 0.62
	- CART 0.80
	- SVM linear 0.60
	- Neural Network MPL 0.58

- for TF vectorisation the best average accuracy scores:
	- LR 0.69
	- KNN 0.40
	- CART 0.62
	- SVM linear 0.67
	- Neural Network MPL 0.62

- the logistic regression and the neural network for TF vectorisation have shown proper label for testing word chains - recommended for future usage.

5. Appendix - search engine & recommender
- to facilitate content management and read articles on certain subject area, simple search engine and recomendation engine was added, both based on cosine similarity.
