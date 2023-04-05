import random
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import apply_features
from nltk import NaiveBayesClassifier
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('movie_reviews')
nltk.download('vader_lexicon')

# Get data of reviews
reviews = [(list(movie_reviews.words(fileid)), category)
  for category in movie_reviews.categories()
  for fileid in movie_reviews.fileids(category)]

# Shuffle reviews to get reviews in random orders
random.shuffle(reviews)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  return features

# Training set
train_set = apply_features(document_features, reviews[:1500])
test_set = apply_features(document_features, reviews[1500:])

# Train classifier
classifier = NaiveBayesClassifier.train(train_set)

# Use a sentiment analyzer to classify sentences
sentiment_analyzer = SentimentIntensityAnalyzer()
print("Rentrez une revue (En anglais)")
sentence = input()
polarity_scores = sentiment_analyzer.polarity_scores(sentence)
print('Précision du model:', nltk.classify.accuracy(classifier, test_set))
if polarity_scores['compound'] > 0:
  print("La revue est positive")
else:
  print("La revue est négative")