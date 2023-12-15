import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
# nltk.download('all')

with open('article1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# translator = str.maketrans("", "", string.punctuation)
# text = text1.translate(translator)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    print("Liczba slow po tokenizacji: ", len(tokens))

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('english')

    filtered_tokens1 = [token for token in tokens if token not in stopwords]
    print("Liczba slow bez stop words: ", len(filtered_tokens1))

    newStopWords = ['.', "''", ',', '``', "'", '…']
    stopwords.extend(newStopWords)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    print("Liczba slow bez stop words Extended: ", len(filtered_tokens), "\n")

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    print("Liczba slow Lemmatize tokens: ", len(lemmatized_tokens))

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

processed_text = preprocess_text(text)
print(processed_text, "\n")



# ############################################################ wykres
def plot_top_words(word_counts):
    # Funkcja do wyświetlenia wykresu słupkowego
    top_words = dict(word_counts.most_common(10))
    fig = plt.figure(figsize=(10, 5))
    plt.bar(top_words.keys(), top_words.values())
    plt.xlabel('Słowa')
    plt.ylabel('Liczba wystąpień')
    plt.title('10 najczęściej występujących słów')
    plt.show()

# Zliczenie słów
word_counts = Counter(processed_text.split())

# Wyświetlenie wykresu
plot_top_words(word_counts)


# ########################################################### chmura tagów
wordcloud = WordCloud().generate(processed_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
