import math
import pandas as pd
import string
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer

qualifiers = {}
modifiers = {}

class BasicInfo:
    def __init__(self):
        self.countries_data = {}
        self.gdp_avg = 0
        self.crime_rate_avg = 0
        self.num_countries = 0
        self.crime_indices = []
        self.initialize_data()

    def initialize_data(self):
        df = pd.read_csv("Datasets/general_information/3. All Countries.csv")
        countries_data = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/general_information/0. Global Country Information Dataset 2023.csv")
        global_info = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/general_information/1. Countries of the World.csv")
        other_info = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/general_information/2. cost_of_living.csv")
        cost_living = df.to_dict(orient='records')
        df = pd.read_csv("Datasets/general_information/4. World Crime Index.csv")
        crime_index = df.to_dict(orient='records')

        for d in countries_data:
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
            for c in global_info:
                if c["Country"].lower() == d["country"].lower() or c["Country"].lower() in d["country"].lower() or d["country"].lower() in c["Country"].lower():
                    d["language"] = c["Language"]
                    break

            j = 0
            d["crime_rate"] = 0
            for c in crime_index:
                if d["country"].lower() in c["City"].lower().rsplit(',', 1)[1]:
                    d["crime_rate"] += c["Crime Index"]
                    j += 1

            if j == 0:
                d["crime_rate"] = -1
            else:
                d["crime_rate"] /= j
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
        
class Country:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

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

def process_text_tags(text):
    tokens = tokenize(text)
    tagged_tokens = pos_tag(tokens)
    tags = [pos for _, pos in tagged_tokens]

    return lemmatize(list(tokens)), tags


def csv_to_asso_arr(path):
    df_words = pd.read_csv(path)
    asso_arr = {}
    for i in range(len(df_words.index)):
        asso_arr[df_words.iat[i, 0]] = df_words.iat[i, 1]
    return asso_arr

def join_strings(strings):
    if strings[len(strings) - 1] == ",":
        strings[len(strings) - 1] = "."
    else:
        strings.append(".")
    if (strings[len(strings) - 3] == ','):
        strings[len(strings) - 3] = " and"
    separator = ""
    return separator.join(strings)

def print_country_info(basic_info, country):
    country_info = basic_info.get_country(country)
    if country_info["gdp"] / country_info["population"] >= basic_info.gdp_avg:
        print("I would recommend you to visit " + country_info["country"] + ", as it is a rich country")
    else:
        print("I wouldn't recommend you to visit " + country_info["country"] + ", as it is a poor country")
    print("It is located in " + country_info["region"] + " and their main language is " + country_info["language"] + ". It has a population of " + str(country_info["population"]) +
                " people, they use the " + country_info["currency"] + " as its currency and its " + country_info["title"].lower() + " is " + country_info["political_leader"])
    
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
            elif unique[2] and (parameter == "danger" or parameter == "secure" or parameter == "safe"):
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

    if (go == 0 and len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        print("I can't understand. Please reformulate or elaborate more your words.")
    elif (len(countries) == 0 and len(continents) == 0 and len(parameters) == 0):
        if is_positive >= 0:
            print("In my opinion the best country to go would be Spain. It has a relatively cheap living cost, has a good service infrastructure and it has a good climate!")
        else:
            print("If I had to say one, I would say Somalia is the worst country to go. It is a very poor country, underdeveloped, with a very high criminal rate and risk of dying!")
    elif (len(countries) == 1):
        check_country(basic_info, countries[0], parameters)
    elif (len(countries) > 1):
        for country in countries:
            check_country(basic_info, country, parameters)
    elif (len(continents) == 1):
        #TODO(For phase 3): same as countries but looking through countries in a continent
        print("Feature yet to implement")
    elif (len(continents) > 1):
        #TODO(For phase 3): same as above for n continents
        print("Feature yet to implement")
    else:
        print("I can't understand. Please reformulate or elaborate more your words.")

def main():

    exit_words = ["exit", "quit", "bye", "goodbye"]
    negative_words = ["worst", "awful",'bad', "terrible"]
    positive_words = ["best","excellent", "amazing", "incredible", "nice", "wonderful"]
    continent_words = ["europe", "asia", "africa", "america", "oceania"]
    # Words that might be used for looking for a country based on a parameter or ask for the information of the country parameter
    visit_words = ["recommend", "go", "visit", "place"]
    key_words = ["currency", "located", "urban", "rural", "develop", "danger", "secure", "safe", "expenses", "rich" , "poor", "information"]

    # Group data into list of dictionaries based on the first letter    
    basic_info = BasicInfo()

    qualifiers = csv_to_asso_arr('Datasets/dictionaries/qualifier_ratings.csv')
    modifiers  = csv_to_asso_arr('Datasets/dictionaries/modifier_ratings.csv')

    while True:    
        # Read and process input (tokenization, filtering and lemmatization)
        data = input("\n> ")
        # Process the text
        processed_data = process_text(data.lower())
        words, tags = process_text_tags(data.lower())
        
        for i in range (len(words)):
            print(words[i], " ", tags[i])

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
    nltk.download("averaged_perceptron_tagger")
    main()
