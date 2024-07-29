import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import pandas as pd

INPUT_FOLDER = "fed_minutes"
KEYWORDS = ['inflation', 'labor', 'growth',
            'unemployment', 'deficit', 'surplus',
            'export', 'import', 'job', 'jobs',
            'consumption', 'investment', 'trade']
sia = SentimentIntensityAnalyzer()   # initialize VADER model

boo = sia.polarity_scores("I like going to the beach.")
print(boo)


def analyze_files():
    pass


def keyword_count():
    pass


def single_meeting_sentiment():
    pass


def keyword_trend_plot():
    pass


def sentiment_plot():
    pass


def make_word_cloud():
    pass
