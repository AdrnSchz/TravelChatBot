import math
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer

class BasicInfo:
    def __init__(self, data):
        self.countries_data = {}
        self.gdp_avg = 0
        self.crime_rate_avg = 0
        self.num_countries = 0
        self.crime_indices = []
        self.initialize_data(data)

    def initialize_data(self, data):
        df = pd.read_csv("Datasets/1. Countries of the World.csv")
        other_info = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/2. cost_of_living.csv")
        cost_living = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/4. World Crime Index.csv")
        crime_index = df.to_dict(orient='records')

        for d in data:
            for c in cost_living:
                if c["country"].lower() == d["country"].lower() or c["country"].lower() in d["country"].lower() or d["country"].lower() in c["country"].lower():
                    d["cost_of_living"] = c["cost_of_living"]
                    d["global_rank"] = c["global_rank"]
                    break
            for c in other_info:
                if c["Country"].lower() == d["country"].lower() or c["Country"].lower() in d["country"].lower() or d["country"].lower() in c["Country"].lower():
                    d["area"] = c["Area"]
                    d["coastline"] = c["Coastline"]
                    d["literacy"] = c["Literacy"]
                    d["phones"] = c["Phones"]
                    d["climate"] = c["Climate"]
                    break
            
            # Extract country's average crime rate with rates of its cities
            i = 0
            d["crime_rate"] = 0
            for c in crime_index:
                if d["country"].lower() in c["City"].lower().rsplit(',', 1)[1]:
                    d["crime_rate"] += c["Crime Index"]
                    i += 1

            if i == 0:
                d["crime_rate"] = -1
            else:
                d["crime_rate"] /= i
                self.crime_indices.append(d["crime_rate"])

            # Countries are grouped by their first letter to allow faster access and search later
            first_letter = d["country"][0]
            if first_letter not in self.countries_data:
                self.countries_data[first_letter] = []
            self.countries_data[first_letter].append(d)

            # Add country's gdp per capita to later calculate average
            if "gdp" in d and not math.isnan(d["gdp"]) and "population" in d and not math.isnan(d["population"]):
                self.gdp_avg += d["gdp"] / d["population"]
                self.num_countries += 1

        self.gdp_avg /= self.num_countries

       # Sort and set crime rate average as the median of the country rates
        self.crime_indices.sort()
        self.crime_rate_avg = self.crime_indices[int(len(self.crime_indices)/2)]
    
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

def join_strings(strings):
    if strings[len(strings) - 1] == ",":
        strings[len(strings) - 1] = "."
    else:
        strings.append(".")
    separator = ""
    return separator.join(strings)

def print_country_info(basic_info, country):
    country_info = basic_info.get_country(country)
    if country_info["gdp"] / country_info["population"] >= basic_info.gdp_avg:
        print("I would recommed you to visit", country_info["country"], ", as it is a rich country")
    else:
        print("I wouldn't recommed you to visit", country_info["country"], ", as it is a poor country")
    print("It is located in", country_info["region"], "and their main language is. It has a population of", country_info["population"], 
                "people, they use the ", country_info["currency"], " as its currency and its ", country_info["title"].lower(), " is ", 
                country_info["political_leader"])
def check_country(basic_info, country, parameters):
    if (len(parameters) == 0):
        print_country_info(basic_info, country)
        country_info = basic_info.get_country(country)
    else:
        unique = [True, True, True, True, True, True, True]
        country_info = basic_info.get_country(country)
        strings = []
        strings.append(country_info["country"])
        coma = True
        for parameter in parameters:
            coma = True

            if unique[0] and parameter == "currency":
                unique[0] = False
                strings.append(" uses " + country_info["currency"] + " as its currency")
            elif unique[1] and (parameter == "rich" or parameter == "poor"):
                unique[1] = False
                if country_info["gdp"] / country_info["population"] >= basic_info.gdp_avg:
                    strings.append(" is a rich country")
                else:
                    strings.append(" is a poor country")
            elif unique[2] and (parameter == "danger" or parameter == "secur" or parameter == "safe"):
                unique[2] = False
                if abs(basic_info.crime_rate_avg - country_info["crime_rate"]) < 10:
                    strings.append(" has a medium crime rate")
                elif country_info["crime_rate"] >= basic_info.crime_rate_avg:
                    strings.append(" has a high crime rate")
                elif country_info["crime_rate"] != -1:
                    strings.append(" has a low crime rate")
                else:
                    strings.append(" does not have enough crime statistics to determine whether it is safe or not")
            else:
                coma = False
            if coma:
                strings.append(",")    
        print(join_strings(strings))    

# Evaluate data taken from the input text
def evaluate_data(basic_info, countries, continents, go, parameters, positive, negative):
    is_positive = positive - negative
    # TODO: Should make a function to grade countries and be able to order them from best to worst

    if (go == 0 and len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        print("I can't understand. Please reformulate or elaborate more your words.")
    elif (len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        if is_positive >= 0:
            print("In my opinion the best country to go would be Spain. It has a relatively cheap living cost, has a good service infrastructure and it has a good climate!")
        else:
            print("If I had to say one, I would say Somalia is the worst country to go. It is a very poor country, underdeveloped, with a very high criminal rate and risk of dying!")
    elif (len(countries) == 1):
        print(check_country(basic_info, countries[0], parameters))
    elif (len(countries) > 1):
        for country in countries:
            print(check_country(basic_info, country, parameters))
    elif (len(continents) == 1):
        #TODO: same as countries but looking through countries in a continent
        print("a")
    elif (len(continents) > 1):
        #TODO: for continent in continents
        print("a")
    else:
        print("I can't understand. Please reformulate or elaborate more your words.")
                     
def main():

    exit_words = ["exit", "quit", "bye", "goodbye"]
    negative_words = ["worst", "awful",'bad', "terrible"]
    positive_words = ["best","excellent", "amazing", "incredible", "nice", "wonderful"]
    continent_words = ["europe", "asia", "africa", "america", "oceania"]
    # Words that might be used for looking for a country based on a parameter or ask for the information of the country parameter
    visit_words = ["recommend", "go", "visit", "place"]
    key_words = ["currency", "located", "urban", "rural", "develop", "danger", "secure", "safe", "expenses", "rich" , "poor", "information", "tell"]
    df = pd.read_csv("Datasets/3. All Countries.csv")
    data = df.to_dict(orient='records')

    # Group data into list of dictionaries based on the first letter    
    basic_info = BasicInfo(data)

    while True:    
        # Read and process input (tokenization, filtering and lemmatization)
        data = input("\n> ")
        # Process the text
        processed_data = process_text(data.lower())

        # Check text
        countries = []
        continents = []
        parameters = []
        positive = 0
        negative = 0
        finish = 0
        go = 0
        for i, word in enumerate(processed_data):
            for d in basic_info.countries_data[word[0].upper()]:
                if d["country"].lower() == word or (word in d["country"].lower() and i+1 < len(processed_data) and processed_data[i+1] in d["country"].lower()):
                    countries.append(d["country"].lower())
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
                if key_word in word:
                    parameters.append(key_word)

            # Last checking to be made
            for exit_word in exit_words:
                if exit_word == word:
                    finish = 1

        if finish == 1:
            print("Bye, have a great day!")
            return
        evaluate_data(basic_info, countries, continents, go, parameters, positive, negative)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()
