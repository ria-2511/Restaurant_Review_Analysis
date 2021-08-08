# Introduction
Using Natural Language Processing and Bag of Words for feature extraction for sentiment analysis of the customers visited in the Restaurant and at last using Classification algorithm to separate Positive and Negative Sentiments. 

## What do you mean by SENTIMENT ANALYSIS?
Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. 
Since customers express their thoughts and feelings more openly than ever before, sentiment analysis is becoming an essential tool to monitor and understand that sentiment. Automatically analyzing customer feedback, such as opinions in survey responses and social media conversations, allows brands to learn what makes customers happy or frustrated, so that they can tailor products and services to meet their customers’ needs.

## Types of Sentiment Analysis 
Here are Some of the Most Popular Types of Sentimeent Analysis - 
### 1. Fine Grained Sentiment Analysis 
If polarity precision is important to your business, you might consider expanding your polarity categories to include:

Very positive
Positive
Neutral
Negative
Very negative
This is usually referred to as fine-grained sentiment analysis, and could be used to interpret 5-star ratings in a review, for example:

Very Positive = 5 stars
Very Negative = 1 star

### 2. Emotion Detection 
This type of sentiment analysis aims to detect emotions, like happiness, frustration, anger, sadness, and so on. Many emotion detection systems use lexicons (i.e. lists of words and the emotions they convey) or complex machine learning algorithms.

One of the downsides of using lexicons is that people express emotions in different ways. Some words that typically express anger, like bad or kill (e.g. your product is so bad or your customer support is killing me) might also express happiness (e.g. this is bad ass or you are killing it).

### 3. Aspect Based Sentiment Analysis 
Usually, when analyzing sentiments of texts, let’s say product reviews, you’ll want to know which particular aspects or features people are mentioning in a positive, neutral, or negative way. That's where aspect-based sentiment analysis can help, for example in this text: "The battery life of this camera is too short", an aspect-based classifier would be able to determine that the sentence expresses a negative opinion about the feature battery life.

### 4. Multilingual Sentiment Analysis 
Multilingual sentiment analysis can be difficult. It involves a lot of preprocessing and resources. Most of these resources are available online (e.g. sentiment lexicons), while others need to be created (e.g. translated corpora or noise detection algorithms), but you’ll need to know how to code to use them.

Alternatively, you could detect language in texts automatically with MonkeyLearn’s language classifier, then train a custom sentiment analysis model to classify texts in the language of your choice.

## Rule Based Sentiment Analysis Approach 
In my project I have used Rule Based Sentiment Analysis as my Approach for applying Sentiment Analysis. 
Usually, a rule-based system uses a set of human-crafted rules to help identify subjectivity, polarity, or the subject of an opinion.

These rules may include various NLP techniques developed in computational linguistics, such as:

Stemming, tokenization, part-of-speech tagging and parsing.
Lexicons (i.e. lists of words and expressions).
Here’s a basic example of how a rule-based system works:

Defines two lists of polarized words (e.g. negative words such as bad, worst, ugly, etc and positive words such as good, best, beautiful, etc).
Counts the number of positive and negative words that appear in a given text.
If the number of positive word appearances is greater than the number of negative word appearances, the system returns a positive sentiment, and vice versa. If the numbers are even, the system will return a neutral sentiment.
Rule-based systems are very naive since they don't take into account how words are combined in a sequence. Of course, more advanced processing techniques can be used, and new rules added to support new expressions and vocabulary. However, adding new rules may affect previous results, and the whole system can get very complex. Since rule-based systems often require fine-tuning and maintenance, they’ll also need regular investments.

### NOTE - 
There are other apporaches that you can also use for Sentiment Analysis , such as - 
1. Automatic: systems rely on machine learning techniques to learn from data.
2. Hybrid systems: combines both rule-based and automatic approaches.

## Sentiment Analysis Python
### Scikit-learn 
SCikit-Learn is the go-to library for machine learning and has useful tools for text vectorization. Training a classifier on top of vectorizations, like frequency or tf-idf text vectorizers is quite straightforward. Scikit-learn has implementations for Support Vector Machines, Naïve Bayes, and Logistic Regression, among others.

### NLTK 
NLTK has been the traditional NLP library for Python. It has an active community and offers the possibility to train machine learning classifiers.

### SpaCy 
Spacy is an NLP library with a growing community. Like NLTK, it provides a strong set of low-level functions for NLP and support for training text classifiers.

### TensorFlow
Tensorflow, developed by Google, provides a low-level set of tools to build and train neural networks. There's also support for text vectorization, both on traditional word frequency and on more advanced through-word embeddings.

### Keras 
Keras provides useful abstractions to work with multiple neural network types, like recurrent neural networks (RNNs) and convolutional neural networks (CNNs) and easily stack layers of neurons. Keras can be run on top of Tensorflow or Theano. It also provides useful tools for text classification.

### PyTorch 
Pytorch is a recent deep learning framework backed by some prestigious organizations like Facebook, Twitter, Nvidia, Salesforce, Stanford University, University of Oxford, and Uber. It has quickly developed a strong community.

# Bibliography 
* https://monkeylearn.com/sentiment-analysis/
* https://en.wikipedia.org/wiki/Sentiment_analysis
* https://brand24.com/blog/sentiment-analysis/
* https://algorithmia.com/blog/introduction-sentiment-analysis
