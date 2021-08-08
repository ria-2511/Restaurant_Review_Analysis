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

## Algorithym Definition 
I have attached the Dataset I have used for this problem in my repository. The sentiment analysis is a classification because the output should be either positive or negative. That is why I tried 2 of the classification algorithms on this data set.
* K-Nearest Neighbours  
* Multinomial Naive Bayes 

### (i) K-Nearest Neighbours 
A **supervised machine learning algorithm** (as opposed to an unsupervised machine learning algorithm) is one that relies on labeled input data to learn a function that produces an appropriate output when given new unlabeled data. 
Supervised machine learning algorithms are used to solve classification or regression problems. A **classification problem** has a discrete value as its output. For example, “likes pineapple on pizza” and “does not like pineapple on pizza” are discrete. There is no middle ground.

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph. 
There are vaious ways to calculate the distance for this Algorithm . Euclidean distance is the most popular distance metric. You can also use Hamming distance, Manhattan distance, Minkowski distance as per your need.
In my Project I have used the MINKOWSKI distance. The Minkowski distance between two variabes X and Y is defined as
(∑i=1n|Xi−Yi|p)1/p
The case where p = 1 is equivalent to the Manhattan distance and the case where p = 2 is equivalent to the Euclidean distance. In my project , p=2 i.e. I have used the Euclidean Distance.

### (ii) Multinomial Naive bayes 
Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.

Naive Bayes classifier is a collection of many algorithms where all the algorithms share one common principle, and that is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature. Naive Bayes is of 3 types :
* Multinomial Naive Bayes : Good for when your features (Categorical or Continuous) describe Discrete Frequecy Counts 
* Bernoulli : Good for making predictions from binary features 
* Gaussian : Good for Making predictions from Normally Distributed Data 

Bayes theorem calculates probability P(c|x) where c is the class of the possible outcomes and x is the given instance which has to be classified, representing some certain features.
***P(c|x) = P(x|c) * P(c) / P(x)***

# Experimental Evaluation
## 3.1 Methodology
All the models were judged based on a few criteria. These criteria are also recommended by the scikit-learn website itself for the classification algorithms. The criteria are:

### Accuracy score: 
Classification Accuracy is what we usually mean when we use the term accuracy. It is the ratio of the number of correct predictions to the total number of input samples.

### Confusion Matrix: 
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. i) There are two possible predicted classes: "yes" and "no". If we were predicting the presence of a disease, for example, "yes" would mean they have the disease, and "no" would mean they don't have the disease. ii) The classifier made a total of 165 predictions (e.g., 165 patients were being tested for the presence of that disease). iii) Out of those 165 cases, the classifier predicted "yes" 110 times, and "no" 55 times. iv) In reality, 105 patients in the sample have the disease, and 60 patients do not.

**true positives (TP)**: These are cases in which we predicted yes (they have the disease), and they do have the disease.
**true negatives (TN)**: We predicted no, and they don't have the disease.
**false positives (FP)**: We predicted yes, but they don't have the disease. (Also known as a "Type I error.")
**false negatives (FN)**: We predicted no, but they do have the disease. (Also known as a "Type II error.")
F1 score F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances). High precision but lower recall, gives you an extremely accurate, but it then misses a large number of instances that are difficult to classify. The greater the F1 Score, the better is the performance of our model. Mathematically, it can be expressed as : F1 Score tries to find the balance between precision and recall.

### Precision: 
It is the number of correct positive results divided by the number of positive results predicted by the classifier.

### Recall: 
It is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).

## 3.2 Result
All of the 2 mentioned machine learning models are very measured on the above-mentioned metrics. The result of the evaluation of the metrics is mentioned below:

### i) Multinomial Naive Bayes:

Accuracy - 0.808

### ii) KNN Classification 

Accuracy - 0.701

# Work 
1. I chose a dataset which contained various Coloumns like Reviews , Reviewer Name , Rating , Time and Restaurant Name. 
2. Then I have Preprocessed this data . A few of the important things I have done while Preprocessing is Dropping the Unnecesary coloumns , checking for any irrelevant values , Replacing the Null values with the median etc. 
3. I have also done some Data Visualization such as plotting graphs on the basis of the Rating Provided. 
4. I used NLTK (Natural Language Toolkit) and cleared the unwanted words in my vector. I accepted only alphabets and converted it into lower case and split it in a list. Using the PorterStemmer method stem I shorten the lookup and Normalized the sentences. Then stored those words which are not a stopword or any English punctuation.
5. Secondly, I used CountVectorizer for vectorization. Also used fit and transform to fit and transform the model. The maximum features were 9000.
6. I have also used the TfidfVectorizer 
7. The next step was Training and Classification. Using train_test_split 30% of data was used for testing and remaining was used for training. 
8. I have used algorithms like KNN classifier and Multinomial Naive Bayes.
9. Later metrics like Confusion matrix, Accuracy, Precision, Recall were used to calculate the performance of the model.
10. Then I have created a function to claissfy a review being passed as it's argument as positve or negative with the help of the bag of words we had created. 

# Scope for Improvement 
* Different classifier models can also be tested.
* Try a different data set. Sometimes a data set plays a crucial role too.
* Some other tuning parameters to improve the accuracy of the model.

# Bibliography 
* https://monkeylearn.com/sentiment-analysis/
* https://en.wikipedia.org/wiki/Sentiment_analysis
* https://brand24.com/blog/sentiment-analysis/
* https://algorithmia.com/blog/introduction-sentiment-analysis
* https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
* https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/minkdist.htm
* https://www.upgrad.com/blog/multinomial-naive-bayes-explained/#:~:text=Multinomial%20Naive%20Bayes%20algorithm%20is,of%20email%20or%20newspaper%20article.
* https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
* https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
