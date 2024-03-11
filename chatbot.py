import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer

# Tokenize and remove punctuation
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [w for w in tokens if not w in string.punctuation]

# Filters out stop words
def filter_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [w for w in tokens if not w.lower() in stop_words]

# Stemming
def stem(tokens):
    porter = PorterStemmer()
    words = []
    for w in tokens:
        words.append(porter.stem(w))
    return words

# Lemmatization
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    words = []
    for w in tokens:
        words.append(lemmatizer.lemmatize(w))
    return words

# Process the text(tokenize, remove punctuation, stop words, and stem/lemmatize)
def process_text(text):
    tokens = tokenize(text)
    filtered = filter_stop_words(tokens)
    return lemmatize(filtered)

# Check words to end the program
def checkExit(text):
    exit_words = ["exit", "quit", "bye", "goodbye"]
    for word in text:
        for exit_word in exit_words:
            if word.lower() == exit_word:
                return True

def main():
    while True:
        # Read the data
        data = input("\n> ")
        # Process the text
        processed_data = process_text(data)
        print(processed_data)

        # Check for exit
        if checkExit(processed_data):
            print("Bye, have a great day!")
            return

if __name__ == "__main__":
    main()
