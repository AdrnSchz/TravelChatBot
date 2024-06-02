import math
import csv
import pandas as pd
import string
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from collections import deque
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer

'''
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
'''
'''
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

def is_country(words, i):

    for j in range(len(all_countries)):
        # Check for multiple word countries (take into account words that have been removed)
        #if words[i][0] in countries_df.iat[j, 0]:
        if (all_countries[j]).lower() == words[i][0]:
            return 1
    return 0

def curr_description_valid(description):
    
    for word in description:
        if 'JJ' in word[1] or 'RB' in word[1] or word[0] == 'compare':
            return True
    return False

def input_to_arrays(words):

    descriptions = []
    description = []
    countrieses = []
    countries = []

    for i, word in enumerate(words):
        # Choosing certain types of words to be filtered out to process less words
        unimportant = ['CC', 'DT', 'EX', 'FW', 'IN', 'MD', 'VBZ']
        if word[1] not in unimportant:
            # Check if a country is being referenced
            country_words = is_country(words, i)
            if country_words != 0 or 'countr' in word[0] or 'nation' in word[0]:
                #print('in first if', word[0])
                if curr_description_valid(description):
                    descriptions.append(description)
                    description = []
                elif not curr_description_valid(description) and len(description) > 0 and len(descriptions) > 0:
                    descriptions[len(descriptions)-1] = descriptions[len(descriptions)-1] + description
                    description = []
                countries.append(word[0])
            # Check if it is type of adjective, adverb or noun
            elif 'JJ' in word[1] or 'RB' in word[1] or 'NN' in word[1] or 'VB' in word[1] or 'CD' == word[1] or word[0] == 'compare':
                #print('in second if', word[0])
                if len(countries) != 0:
                    countrieses.append(countries)
                    countries = []
                description.append(word)
            #else:
                #print('discarded in second if:', word[0], word[1])
        #else:
            #print ('discarded in first if:', word[0])
    if len(countries) > 0:
        countrieses.append(countries)
    if curr_description_valid(description):
        descriptions.append(description)
    elif not curr_description_valid(description) and len(description) > 0 and len(descriptions) > 0:
        descriptions[len(descriptions)-1] = descriptions[len(descriptions)-1] + description
    elif len(descriptions) == 0:
        descriptions.append(description)

    return countrieses, descriptions
'''

qualifiers = {}
modifiers = {}
keys = []
porter = PorterStemmer()
countries_df = pd.DataFrame()
all_countries = []

def txt_to_csv_column(txt_name):

    target_column_index = 0
    path_split = (txt_name.lower()).split('.')
    column_name = (path_split[0]).capitalize()
    txt_path = 'Datasets/country_ratings/' + txt_name

    ratings = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line_split = line.split(' - ')
        ratings.append(line_split[1].replace('\r', '').replace('\n', ''))

    csv_path = 'Datasets/country_attributes.csv'
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        try:
            target_column_index = header.index(column_name)
        except ValueError:
            print('no column found with the name', column_name)
            return

    rows = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for i in range(len(ratings)):
        rows[i+1][target_column_index] =  ratings[i]

    try:
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    except:
        print('unable to write in', csv_path)

def countries(country, attribute):
    return countries_df.at[country, attribute]

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
    return pos_tag(tokens)

def csv_to_asso_arr(path):
    df_words = pd.read_csv(path)
    asso_arr = {}
    for i in range(len(df_words.index)):
        asso_arr[df_words.iat[i, 0]] = df_words.iat[i, 1]
    return asso_arr

def load_messages(file_path):
    msgs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, message = line.strip().split('=')
            msgs[key] = message
    return msgs

def is_country(words, i):
    # TODO: Check for country names with more than one word (take into account words that have been removed)
    # TODO: Check for variations of a country name
    for j in range(len(all_countries)):
        
        #if words[i][0] in countries_df.iat[j, 0]:
        if (all_countries[j]).lower() == words[i][0]:
            return 1
    return 0

def add_pot_attr(attributes, word):

    value = modifiers.get(word, -1000)
    tag = 'MDF'
    if value == -1000:
        value = qualifiers.get(word, -1000)
        tag = 'QLF'
        if value == -1000:
            with open('Datasets/dictionaries/attribute_synonyms.txt', 'r') as file:
                lines = file.readlines()

            for line in lines:
                line = line.replace('\n', '').replace('\r', '')
                attr_synonyms = line.split(', ')

                for i in range(len(attr_synonyms)):
                    if word == attr_synonyms[i] or porter.stem(word) == porter.stem(attr_synonyms[i]):
                        word = attr_synonyms[0]
                        value = 0
                        tag = 'ATR'
                        break
                if value == 0:
                    break

            if value == -1000:
                return False
    
    tuple = [word, value, tag]
    attributes.append(tuple)
    return True


def input_to_arrays(words):

    #continents = []
    countries = []
    attributes = []
    description = []

    for i, word in enumerate(words):
        # Choosing certain types of words to be filtered out to process less words
        unimportant = ['CC', 'DT', 'EX', 'FW', 'IN', 'MD', 'VBZ', 'WDT', 'WP', 'WP$', 'WRP']
        if word[1] not in unimportant:
            # Check if a country is being referenced
            country_words = 0
            country_words = is_country(words, i)
            if country_words > 0:
                countries.append(word[0])
            elif 'countr' in word[0] or 'nation' in word[0] or 'plac' in word[0] or 'dest' in word[0] or 'locat' in word[0]:
                countries.append('country')
            elif 'JJ' in word[1] or 'RB' in word[1] or 'NN' in word[1] or 'VB' in word[1]:
                if not add_pot_attr(attributes, word[0]):
                    description.append(word)
            else:
                description.append(word)

    return countries, attributes, description

def print_inlist_format(list):
    for i, object in enumerate(list):
        print(object, end='')
        if i + 1 == len(list):
            print(' ', end='')
        elif i + 2 == len(list):
            print(' and ', end='')
        else:
            print(', ', end='')
    return

def attribute_comparison(countries, attributes, comparison, thereis_attr):
    if not thereis_attr:
        attributes = []
        for attribute in countries_df.columns:
            attributes.append([attribute, 0, 'ATR'])

    no_actual_country = True
    for country in countries:
        if country != 'country':
            no_actual_country = False
    
    if no_actual_country:
        countries = list(countries_df.index)

    countries_best = []
    best = 0
    attribute_winners = {attribute[0]: [] for attribute in attributes if attribute[2] == 'ATR'}

    for country in countries:
        if country != 'country':
            i = 0
            average_attributes_rating = 0.0

            for i, attribute in enumerate(attributes):
                if attribute[2] == 'ATR':
                    if attribute[0] != 'temperature':
                        average_attributes_rating += countries_df.loc[country, attribute[0]]
                        if not attribute_winners[attribute[0]] or countries_df.loc[country, attribute[0]] > countries_df.loc[attribute_winners[attribute[0]][0], attribute[0]]:
                            attribute_winners[attribute[0]] = [country]
                        elif countries_df.loc[country, attribute[0]] == countries_df.loc[attribute_winners[attribute[0]][0], attribute[0]]:
                            attribute_winners[attribute[0]].append(country)
                    else:
                        # TODO: handle temperature
                        i -= 1
                
            average_attributes_rating = (average_attributes_rating / (i+1))

            if average_attributes_rating > best:
                countries_best = [country]
                best = average_attributes_rating
            elif average_attributes_rating == best:
                countries_best.append(country)

    lastWinner = ''

    for attribute, winners in attribute_winners.items():
        again = False
        different = False

        if len(winners) == 1 and winners[0] == lastWinner:
            again = True
        elif len(winners) == 1 or (lastWinner != '' and lastWinner not in winners):
            different = True

        if different:
            print(f"Nevertheless, in terms of " + attribute + ", ", end='')
        else:
            print("In terms of" + attribute + ", ", end='')

        print_inlist_format(winners)
        if again:
            print(' is also the best ', end='')
        elif (len(winners) > 1):
            lastWinner = ''
            print(' are the best ', end='')
        else:
            lastWinner = winners[0]
            print('is the best ', end='')


    if len(countries_best) == len(countries):
        print('Overall, all the countries you mentioned are similarly good. The best one will depend on your preferences.')
    elif len(countries_best) > 1:
        print('Overall, ', end='')
        print_inlist_format(countries_best)
        print('are similarly good. The best one will depend on your preferences.')
    else:
        print(attribute_names)
        print(countries_best[0]+' has the '+comparison+' ', end='')
        print_inlist_format(attribute_names)


def process_input(countries, attributes, description):

    thereis_attr = False
    for word in attributes:
        if word[2] == 'ATR':
            thereis_attr = True

    comparison = ''
    for word in description:
        if word[1] == 'RBR' or word[1] == 'JJR' or word[1] == 'RBS' or word[1] == 'JJS' or word[0] == 'nicer' or word[0] == 'compare':
            comparison = word[0]
        

    if comparison != '' and len(countries) > 0:
        attribute_comparison(countries, attributes, comparison, thereis_attr)        
    elif len(countries) > 0:
        for country in countries:
            check_country(country, attributes)
    else:
        print('I can\'t understand. Please reformulate or elaborate more your words.')
    return

def main():

    txt_to_csv_column('coast.txt')
    txt_to_csv_column('culture.txt')
    txt_to_csv_column('expense.txt')
    txt_to_csv_column('gastronomy.txt')
    txt_to_csv_column('monuments.txt')
    txt_to_csv_column('nature.txt')
    txt_to_csv_column('nightlife.txt')
    txt_to_csv_column('safety.txt')
    txt_to_csv_column('shopping.txt')
    txt_to_csv_column('skiing.txt')
    txt_to_csv_column('temperature.txt')
    txt_to_csv_column('tourism.txt')
    global countries_df
    countries_df = pd.read_csv('Datasets/country_attributes.csv', index_col=0)
    countries_df = countries_df.dropna()
    countries_df.columns = countries_df.columns.str.lower()
    countries_df.index = countries_df.index.str.lower()

    global all_countries
    all_countries = countries_df.index

    continent_words = ["europe", "asia", "africa", "america", "oceania"]

    global qualifiers
    qualifiers = csv_to_asso_arr('Datasets/dictionaries/qualifier_ratings.csv')
    global modifiers
    modifiers  = csv_to_asso_arr('Datasets/dictionaries/modifier_ratings.csv')

    global messages
    messages = load_messages('Datasets/messages.txt')
    while True:    
        # Read and process input (tokenization, filtering and lemmatization)
        data = input("\n> ")
        # Process the text
        words = process_text_tags(data.lower())

        print(words)
        print()

        #continents = []
        countries = []
        attributes = []
        description = []

        countries, attributes, description = input_to_arrays(words)
        print()
        print(countries)
        print(attributes)
        print(description)

        process_input(countries, attributes, description)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download("averaged_perceptron_tagger")
    main()
