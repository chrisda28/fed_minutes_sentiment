import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()

boo = sia.polarity_scores("I like going to the beach.")
print(boo)
