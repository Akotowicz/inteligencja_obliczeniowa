import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import text2emotion as te

with open('OpinionP.txt', 'r', encoding='utf-8') as file:
    opinion_p = file.read()
with open('OpinionN.txt', 'r', encoding='utf-8') as file:
    opinion_n = file.read()

def preprocess_text(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text.lower())
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopwords]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def test_opinions(opinion):
    sid = SentimentIntensityAnalyzer()
    print(opinion)
    print(sid.polarity_scores(preprocess_text(opinion)))
    # print(te.get_emotion(opinion))
    print("")

test_opinions(opinion_p)
test_opinions(opinion_n)
test_opinions("this is shitty place, i hated this")