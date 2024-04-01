import math
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer

class Basic_info:
    def __init__(self, data):
        self.countries_data = {}
        self.gdp_avg = 0
        self.num_countries = 0
        self.initialize_data(data)

    def initialize_data(self, data):
        for d in data:
            first_letter = d["country"][0]
            if first_letter not in self.countries_data:
                self.countries_data[first_letter] = []
            self.countries_data[first_letter].append(d)
            if "gdp" in d and not math.isnan(d["gdp"]):
                self.gdp_avg += d["gdp"]
                self.num_countries += 1
            
        self.gdp_avg = self.gdp_avg / self.num_countries
    
    def get_country(self, country):
        for initial_letter, countries_list in self.countries_data.items():
            for country_info in countries_list:
                if country_info["country"].lower() == country:
                    return country_info
        

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
    # If filter stop words, it cannot recognize a question
    filtered = filter_stop_words(tokens)
    return lemmatize(filtered)

def check_country(basic_info, country, parameters):
    if (len(parameters) == 0):
        country_info = basic_info.get_country(country)
        if country_info["gdp"] >= basic_info.gdp_avg:
            print("I would recommed you to visit", country_info["country"], ", as it is a rich country and...")
        else:
            print("I wouldn't recommed you to visit", country_info["country"], ", as it is a poor country and...")

# Evaluate data taken from the input text
def evaluate_data(basic_info, countries, continents, go, parameters, positive, negative):
    is_positive = positive - negative
    # Should make a function to grade countries and be able to order them from best to worst

    if (go == 0 and len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        print("Please explain better.")
    elif (len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        if is_positive >= 0:
            print("In my opinion the best country to go would be Spain. As it has a cheap living cost, has a good service infrastructure and it has a good climate!")
        else:
            print("If I had to say one, I would say Somalia is the worst country to go. As it is a very poor country, underdeveloped, with a very high criminal rate and risk of dying!")    
    elif (len(countries) == 1):
        print(check_country(basic_info, countries[0], parameters))
    elif (len(countries) > 1):
        for country in countries:
            print(check_country(basic_info, country, parameters))
    elif (len(continents) == 1):
        # same as countries but looking through countries in a continent
        print("texto de ejemplo")
    elif (len(continents) > 1):
        for continent in continents: 
            print("texto de ejemplo")      
    else:
        print("Please develop more")
                     
def main():
    exit_words = ["exit", "quit", "bye", "goodbye"]
    negative_words = ["worst", "awful",'bad', "terrible"]
    positive_words = ["best","excellent", "amazing", "incredible", "nice", "wonderful"]
    continent_words = ["europe", "asia", "africa", "america", "oceania"]
    # Words that might be used for looking for a country based on a parameter or ask for the information of the country parameter
    visit_words = ["recommend", "go", "visit", "place"]
    key_words = ["currency", "located", "urban", "rural", "develop", "danger", "secure", "safe", "expenses", "rich" , "poor"]
    df = pd.read_csv("Datasets/3. All Countries.csv")
    data = df.to_dict(orient='records')
    
    # Group data into list of dictionaries based on the first letter    
    basic_info = Basic_info(data)

    while True:    
        # Read the data
        data = input("\n> ")
        # Process the text
        processed_data = process_text(data.lower())
        print(processed_data)

        # Check text
        countries = []
        continents = []
        parameters = []
        positive = 0
        negative = 0
        finish = 0
        go = 0
        for word in processed_data:
            for d in basic_info.countries_data[word[0].upper()]:
                if d["country"].lower() == word or word in d["country"].lower():
                    countries.append(word)

            for positive_word in positive_words:
                if positive_word == word:
                    positive += 1

            for negative_word in negative_words:
                if negative_word == word:
                    negative += 1

            for continent_word in continent_words:
                if continent_word == word:
                    continents.append(word)
            
            for visit_word in visit_words:
                if visit_word == word:
                    go = 1

            for key_word in key_words:
                if key_word == word:
                    parameters.append(word)

            # Last checking to be made
            for exit_word in exit_words:
                if exit_word == word:
                    finish = 1

        evaluate_data(basic_info, countries, continents, go, parameters, positive, negative)
        if finish == 1:
            print("Bye, have a great day!")
            return

if __name__ == "__main__":
    main()
