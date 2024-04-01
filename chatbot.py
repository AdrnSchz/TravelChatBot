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

def main():
    exit_words = ["exit", "quit", "bye", "goodbye"]
    negative_words = ["worst", "awful",'bad', "terrible"]
    positive_words = ["best","excellent", "amazing", "incredible", "nice", "wonderful"]
    continent_words = ["europe", "asia", "africa", "america", "oceania"]
    # Words that might be used for looking for a country based on a parameter or ask for the information of the country parameter
    key_words = ["currency", "place", "locat", "urban", "rural", "develop", "danger", "secure", "safe", "expenses", "rich" , "poor"]
    df = pd.read_csv("Datasets/3. All Countries.csv")
    data = df.to_dict(orient='records')
    
    # Group data into list of dictionaries based on the first letter
    countries_data = {}
    for d in data:
        first_letter = d["country"][0]
        if first_letter not in countries_data:
            countries_data[first_letter] = []
        countries_data[first_letter].append(d)

    while True:
        # Read the data
        data = input("\n> ")
        # Process the text
        processed_data = process_text(data)
        print(processed_data)

        # Check text
        countries = []
        continents = []
        positive = 0
        negative = 0
        for word in text:
            for d in countries_data[word[0].upper()]:
                if word.lower() == d["country"].lower() or word.lower() in d["country"].lower():
                    countries.append(word)

            for positive_word in positive_words:
                if positive_word == word.lower():
                    positive += 1

            for negative_word in negative_words:
                if negative_word == word.lower():
                    negative += 1

            # Last checking to be made
            for exit_word in exit_words:
                if word.lower() == exit_word:
                    print("Bye, have a great day!")
                    return

if __name__ == "__main__":
    main()
