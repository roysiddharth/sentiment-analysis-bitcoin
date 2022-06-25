# Sentiment Analysis of news headlines to predict Bitcoin market fluctuations
NLP and Machine Learning project to predict the Bitcoin price fluctuations by performing sentiment analysis of news headlines.

# Author
[Siddhartha Roy](https://pages.github.com/roysiddharth)

# Introduction
In the past decade, another class of financial assets, cryptocurrency, arose into the asset management
industry in risky portfolios, because of their low correlations with other major financial assets. As per
Coinmarketcap, the cryptocurrency market has evolved dramatically over the past few years, with market
capitalization surpassing US$2tn. As of April 21, 2021, the Bitcoin (BTC) market value already surpassed
US$1tn with 89% of coins in circulation. CryptoCoinCharts shows 10,125 crypto coins as of April 2021,
mostly attributed to factors like BTC open source which allows the continual creation of new
cryptocurrencies.
Natural Language Processing (NLP), as an emerging branch of ML, has particularly seen various
applications in finance and asset management. Sentiment Analysis is one such area where attempts are
made to uncover information from content like financial news headlines. For instance, it was found in
previous studies that user comments in online crypto communities can predict changes in BTC prices.
Some studies also provide evidence to show that BTC is more of a speculative bubble whose prices and
volume of trade depend on people’s sentiments about it.
This study that we have conducted aims to explore the unexplored seas of crypto market volatilities using
methods of text and sentiment analysis to predict whether the prices of BTC increased or decreased on a
given day.

# Libraries
- Pandas
- NLTK
- Regular Expressions
- Contractions
- String
- Numpy
- Scipy.stats
- SentimentIntensityAnalyzer (VADER)
- TextBlob
- Scikit-learn
- BeautifulSoup4
- requests

# Features
- **Polarity** - For ascertaining the polarity of a text, the polarity score of each word of the text, if present in the
dictionary, is added to get an 'overall polarity score'. For instance, in the event that a vocabulary matches a
word set apart as certain in the dictionary, the absolute polarity score of the text is increased. We have found the polarity score using the get ‘compound’ function in VADER to get the overall polarity.
- **Fear and Greed Index** - The Fear & Greed Index is a compilation of seven different indicators that measure some aspects of stock
market behavior. They are market momentum, stock price strength, stock price breadth, put and call
options, junk bond demand, market volatility, and haven demand. The index tracks how much these
individual indicators deviate from their averages compared to how much they normally diverge. The
index gives each indicator equal weighting in calculating a score from 0 to 100, with 100 representing
maximum greediness and 0 signaling maximum fear.

# Hyperparameter Tuning

## Pruning
The first step to hyperparameter optimization that we have resorted to is called pruning the decision trees
and random forest classifiers for the best value of “ccp_alpha”. Decision Trees are infamous as they can
cling too much to the data they are trained on. This leads to poor deployment because it cannot deal with
new sets of values. Thus, pruning is done to overcome the problem of overfitting.
When the alpha values are set to zero (default), both the decision tree and the random forest overfits, As
the alpha value is slightly increased, more of the tree is pruned thus creating a decision tree and random
forest that generalizes better. This is especially of importance to us as our data set is of relatively smaller
size. Thus, we need our model to generalize results and predictions better in order to be able to predict
unknown results with maximum accuracy.

## GridSearchCV
We run 4 models through our GridSearchCV: Support Vector Machine, Logistic Regression, Decision
Tree, Random Forest. We define a dictionary with all the necessary hyperparameters for the above
models. We run the cross-validation for 100 iterations and obtain the best estimators as the model
RandomForestClassifier(n_estimators=100, criterion=’entropy’, ccp_alpha=0.0035).

# Conclusion
The shortcomings in the cryptocurrency market persuaded this review to investigate the social part of
their price discovery process. In our study, we have used an independently web scraped text data set to
capture various sentiment analysis measures such as polarity, subjectivity, and market sentiment measures
such as the Fear and Greed Index. The result of the study found that we can predict the fluctuations
(whether the price of BTC rose or fell from the previous) day based on the market sentiment measures
mentioned above with an accuracy of at least 70%. This proves that Bitcoin sentiment has a positive
impact on Bitcoin returns supporting our hypothesis that behavioral aspects play a significant role in the
prices of BTC.

