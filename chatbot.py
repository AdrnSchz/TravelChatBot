import math
import csv
import pandas as pd
import string
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from collections import deque
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import scrolledtext
import io
import sys
import os

class TravelChatbotUI:

    def __init__(self, widget_root):

        self.widget_root = widget_root
        self.widget_root.title("TravelChatbot")

        self.chat = scrolledtext.ScrolledText(widget_root, wrap=tk.WORD, width=30, height=20, state='disabled')
        self.chat.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_box = tk.Entry(widget_root, width=40)
        self.input_box.grid(row=1, column=0, padx=10, pady=10)
        self.input_box.bind("<Return>", self.send_input)

        self.button = tk.Button(widget_root, text="Send", command=self.send_input)
        self.button.grid(row=1, column=1, padx=10, pady=10)

    def send_input(self, event=None):

        input = self.input_box.get()
        if input.strip():
            self.chat.configure(state='normal')
            self.chat.insert(tk.END, f"You: {input}\n")
            self.chat.insert(tk.END, f"Bot: {self.fetch_response(input)}\n")
            self.chat.configure(state='disabled')
            self.chat.yview(tk.END)
            self.input_box.delete(0, tk.END)

    def fetch_response(self, input):

        output_capture = io.StringIO()
        sys.stdout = output_capture

        words = process_text_tags(input.lower())

        countries = []
        attributes = []
        description = []

        countries, attributes, description = input_to_arrays(words)

        process_input(countries, attributes, description)

        captured_output = output_capture.getvalue()

        output_capture.close()

        sys.stdout = sys.__stdout__

        return captured_output

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

    pot_countries = all_countries.copy()
    while (True):
        new_pot_countries = []
        
        for j in range(len(pot_countries)):
            pot_country = (pot_countries[j]).split(' ')
            
            for k in range(i, len(words)):
                if k-i == len(pot_country):
                    break
                if words[k][0] == pot_country[k-i] and pot_countries[j] not in new_pot_countries:
                    #print(words[k][0], pot_country[k-i], 'eq', words[i][0], pot_countries[j], '  ', new_pot_countries, k, i, len(words), len(pot_country))
                    new_pot_countries.append(pot_countries[j])
                else:
                    if words[k][0] != pot_country[k-i]:
                        try:
                            new_pot_countries.remove(pot_countries[j])
                        except Exception:
                            pass
                        break

        pot_countries = new_pot_countries.copy()

        if len(pot_countries) == 0:
            return '', 0
        elif len(pot_countries) == 1:
            return pot_countries[0], len(pot_countries[0])

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

    countries = []
    attributes = []
    description = []

    prev_country = ''
    unimportant = ['CC', 'DT', 'EX', 'FW', 'IN', 'MD', 'WDT', 'WP', 'WP$', 'WRP']

    for i, word in enumerate(words):

        if word[0] not in prev_country:
            # Choosing certain types of words to be filtered out to process less words
            if word[1] not in unimportant:
                # Check if a country is being referenced
                prev_country, country_words = is_country(words, i)
                if country_words > 0:
                    countries.append(prev_country)
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
    else: 
        if 'country' in countries:
            countries.remove('country')

    countries_best = []
    best = 0
    attribute_winners = {attribute[0]: [] for attribute in attributes if attribute[2] == 'ATR'}

    for country in countries:
        if country != 'country':
            i = 0
            average_attributes_rating = 0.0

            for i, attribute in enumerate(attributes):
                if attribute[2] == 'ATR':
                    average_attributes_rating += countries_df.loc[country, attribute[0]]
                    if not attribute_winners[attribute[0]] or countries_df.loc[country, attribute[0]] > countries_df.loc[attribute_winners[attribute[0]][0], attribute[0]]:
                        attribute_winners[attribute[0]] = [country]
                    elif countries_df.loc[country, attribute[0]] == countries_df.loc[attribute_winners[attribute[0]][0], attribute[0]]:
                        attribute_winners[attribute[0]].append(country)
                
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
        elif lastWinner != '' and (len(winners) == 1 or lastWinner not in winners):
            different = True

        if different:
            print(f"Nevertheless, in terms of " + attribute + ", ", end='')
        else:
            print("In terms of " + attribute + ", ", end='')

        print_inlist_format(winners)
        if again:
            print('is also the best. ', end='')
        elif (len(winners) > 1):
            lastWinner = ''
            print('are the best. ', end='')
        else:
            lastWinner = winners[0]
            print('is the best. ', end='')


    if len(countries_best) == len(countries):
        print('\nOverall, all the countries you mentioned are similarly good. The best one will depend on your preferences.')
    elif len(countries_best) > 1:
        print('\nOverall, ', end='')
        print_inlist_format(countries_best)
        print('are similarly good. The best one will depend on your preferences.')
    else:
        print('\nOverall, ', end='')
        print(countries_best[0], end=' ')
        if len(countries) > 2:
            print('is the best. Though depending on how you value each factor, you might find another country as the best.')
        else:
            print('is the best. Though depending on how you value each factor, you might find the other as the best.')

def get_attribute_message(attributes, rating):
    i = 0
    for attribute in attributes:
        key = f"{attribute}_{rating.replace(' ', '_')}"
        msg = messages.get(key, f"No message for {attribute} with rating {rating}")
        if i == 0:
            msg = msg.capitalize()
            print(msg, end='')
        elif i + 1 == len(attributes):
            print(' and ' + msg + '.', end='')
        else:
            print(', ' + msg, end='')
        i += 1

def check_country_attr(country, attributes):
    
    country_info = countries_df.loc[country]
    evaluations = {
        'perfect': [],
        'excellent': [],
        'very good': [],
        'good': [],
        'okay': [],
        'poor': [],
        'very poor': []
    }

    country = country.capitalize()

    average = 0.0
    for attribute in attributes:
        attr_name = attribute[0]
        if attr_name in country_info:
            attr_value = country_info[attr_name]
            average += attr_value
            if attr_value >= 1:
                evaluations['perfect'].append(attr_name)
            if attr_value >= 0.85:
                evaluations['excellent'].append(attr_name)
            elif attr_value >= 0.7:
                evaluations['very good'].append(attr_name)
            elif attr_value >= 0.55:
                evaluations['good'].append(attr_name)
            elif attr_value >= 0.4:
                evaluations['okay'].append(attr_name)
            elif attr_value >= 0.25:
                evaluations['poor'].append(attr_name)
            else:
                evaluations['very poor'].append(attr_name)

    average = average / len(attributes)
    
    first = True
    for rating, attrs in evaluations.items():

        if attrs:
            if first:
                print(f"When talking about {country}, regarding the ", end='')
                first = False
            else:
                print(" In terms of ", end='')
            
            print_inlist_format(attrs)

            if rating == 'perfect':
                print(', it could not be better. ', end='')
            elif rating == 'excellent':
                print(', it\'s excellent. ', end='')
            elif rating == 'very good':
                print(', it\'s very good. ', end='')
            elif rating == 'good':
                print(', it\'s good. ', end='')
            elif rating == 'okay':
                print(', it\'s okay. ', end='')
            elif rating == 'poor':
                print(', it\'s bad. ', end='')
            elif rating == 'very poor':
                print(', it couldn\'t be worse. ', end='')
            get_attribute_message(attrs, rating)

    print_overall(country, average)

def check_country(country, description):

    specific_data = print_country_info(country, description)
    
    if not specific_data:
        country_info = countries_df.loc[country]
        attributes = []

        country = country.capitalize()

        with open('Datasets/dictionaries/attribute_synonyms.txt', 'r') as file:
            lines = file.readlines()

            for line in lines:
                line = line.replace('\n', '').replace('\r', '')
                attr_synonyms = line.split(', ')

                attributes.append([attr_synonyms[0], 0, 'ATR'])         

        print(country, "has the following average ratings:\n")

        average = 0.0
        for attribute in attributes:
            attr_name = attribute[0]
            if attr_name in country_info:
                attr_value = country_info[attr_name]
                average += attr_value
                print("{0}: {1}/10".format(attr_name, int(attr_value * 10)))

        average = average / len(attributes)

        print_overall(country, average)

def print_overall(country, average):
    if (average >= 1):
        print('\nThere is no better place for what you are looking for. You better visit ' + country, end='')
        print('. I bet it will be the best trip of your life!\n')
    elif (average >= 0.85):
        print('\nOverall, I would highly recommend you to visit '+ country, end='')
        print('. You will have an amazing time there!\n')
    elif (average >= 0.5):
        print('\nOverall, ' + country, end='')
        print(' is a good place for you to visit. You will have a great time there!\n')
    elif (average >= 0.25):
        print('\nOverall, I would not recommend you to visit ' + country, end='')
        print('. As it will be lacking in some aspects you are looking for. I am sure there is a better place for you to visit!\n')
    else:
        print('\nI doubt there is a worst place than ' + country, end='')
        print(' for you to visit, anywhere will be better than that place. I would recommend you to avoid it at all costs!\n')

def join_strings(str):
    if str[len(str) - 1] == ",":
        str[len(str) - 1] = "."
    else:
        str.append(".")
    if (str[len(str) - 3] == ','):
        str[len(str) - 3] = " and"
    separator = ""
    return separator.join(str)

def print_basic_info(country):
    country_info = general_data.get_country(country)
    if country_info["gdp"] / country_info["population"] >= general_data.gdp_avg:
        print("As general information, " + country_info["country"] + " is a rich country")
    else:
        print("As general information, " + country_info["country"] + " is a poor country")

    print("It is located in " + country_info["region"] + " and their main language is " + country_info["language"] + ". It has a population of " + str(country_info["population"]) +
                " people, they use the " + country_info["currency"] + " as its currency and its " + country_info["title"].lower() + " is " + country_info["political_leader"] + "\n")
    
def print_country_info(country, parameters):

    if (len(parameters) == 0):
        print_basic_info(country)
        return False
    else:
        unique = [True, True, True, True, True, True, True]
        country_info = general_data.get_country(country)
        strings = []
        strings.append(country_info["country"])
        coma = True
        for parameter in parameters:
            parameter = parameter[0]
            coma = True

            if unique[0] and parameter == "currency":
                unique[0] = False
                strings.append(" uses " + country_info["currency"] + " as its currency")
            elif unique[1] and (parameter == "rich" or parameter == "poor"):
                unique[1] = False
                if country_info["gdp"] / country_info["population"] >= general_data.gdp_avg:
                    strings.append(" is a rich country, it has a gross domestic product of " + str(int(country_info["gdp"])))
                else:
                    strings.append(" is a poor country, it has a gross domestic product of " + str(int(country_info["gdp"])))
            else:
                coma = False
            if coma:
                strings.append(",")  

        if len(strings) == 1:
            print_basic_info(country)
            return False 
                 
        print(join_strings(strings))  
        return True

def get_countries_attr(attributes):

    attr = [attribute[0] for attribute in attributes if attribute[0] in countries_df.iloc[0]]
    
    df_attr = countries_df[attr].copy(True)
    df_attr['score'] = df_attr.sum(axis=1)
    top_countries = df_attr.nlargest(3, 'score').index.tolist()
    
    print("The countries with the best {} are {}.".format(', '.join(attr), ', '.join(top_countries[:3])))

def get_similar_country(country, attributes):

    attr = [attribute[0] for attribute in attributes if attribute[0] in countries_df.iloc[0]]

    whole_comparison = False
    if len(attr) == 0:
        whole_comparison = True
        attr = countries_df.columns.tolist()

    df_attr = countries_df[attr].copy(True)

    # Normalize temperature
    if 'temperature' in attr:
        df_attr['temperature'] = (df_attr['temperature_raw'] - df_attr['temperature_raw'].min()) / (df_attr['temperature_raw'].max() - df_attr['temperature_raw'].min())
        df_attr = df_attr.drop(columns='temperature_raw')

    kmeans = KMeans(5, init='k-means++', n_init=100)
    labels = kmeans.fit_predict(df_attr)

    df_attr['cluster'] = labels
    
    country_cluster = df_attr.loc[df_attr.index == country, 'cluster'].iloc[0]

    similar_countries = df_attr.loc[df_attr['cluster'] == country_cluster]
    similar_countries = similar_countries.drop(country)

    if whole_comparison:
        print("Some countries that are similar to {} are {}.".format(country.capitalize(), ', '.join(similar_countries.sample(n=3).index)))
    else:
        print("Some countries that are similar to {} when it comes to {} are {}.".format(country.capitalize(), ', '.join(attr), ', '.join(similar_countries.sample(n=3).index)))

def check_for_similar(description):
    synonyms = ['similar', 'alternative', 'equivalent', 'equal', 'resembles', 'same']
    return any(word in synonyms for word, _ in description)

def process_input(countries, attributes, description):

    thereis_attr = False

    for word in attributes:
        if word[2] == 'ATR':
            thereis_attr = True

    comparison = ''
    for word in description:
        if word[1] == 'RBR' or word[1] == 'JJR' or word[1] == 'RBS' or word[1] == 'JJS' or word[0] == 'nicer' or word[0] == 'compare':
            comparison = word[0]

    if comparison != '' and (len(countries) == 0 or len(countries) > 2 or (len(countries) == 2 and 'country' not in countries)):
        attribute_comparison(countries, attributes, comparison, thereis_attr)  
    elif any(item != "country" for item in countries) and check_for_similar(description):
        for country in countries:
            if country != "country":
                get_similar_country(country, attributes)
                break
    elif any(item != "country" for item in countries) and len(attributes) > 0:
        for country in countries:
            if country != "country":
                check_country_attr(country, attributes)
    elif "country" in countries and len(attributes) > 0:
        get_countries_attr(attributes)         
    elif any(item != "country" for item in countries):
        for country in countries:
            if country != "country":
                check_country(country, description)
    else:
        print('I can\'t understand. Please reformulate or elaborate more your words.')

class GeneralData:
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
                if country_info["country"].lower() == country.lower():
                    return country_info

def init():

    global general_data
    general_data = GeneralData()

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

    countries_df['temperature_raw'] = countries_df['temperature']

    # Normalize temperature with 20ºC as ideal temperature, apply extra penalization for each degree >= 25ºC or <= 10ºC
    countries_df['temperature'] = countries_df['temperature'].apply(lambda x: min(1, 1 - ((abs(x - 20) + max(0, x - 24) + max(0, 11 - x)) / 20)))

    global all_countries
    all_countries = list(countries_df.index)

    continent_words = ["europe", "asia", "africa", "america", "oceania"]

    global qualifiers
    qualifiers = csv_to_asso_arr('Datasets/dictionaries/qualifier_ratings.csv')
    global modifiers
    modifiers  = csv_to_asso_arr('Datasets/dictionaries/modifier_ratings.csv')

    global messages
    messages = load_messages('Datasets/messages.txt')

if __name__ == "__main__":

    # Avoid known memory leak on Kmeans execution
    os.environ['OMP_NUM_THREADS'] = '1'

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download("averaged_perceptron_tagger")

    init()

    window = tk.Tk()
    app = TravelChatbotUI(window)
    window.mainloop()