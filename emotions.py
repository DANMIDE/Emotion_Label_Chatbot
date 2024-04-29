import nltk
import streamlit as st 
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('emotions.txt', sep =';',header = None, names=['text', 'label'])


#THE CODE FOR PREPROCESSING 
# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['text'].apply(preprocess_text)


xtrain = data['tokenized Questions'].to_list()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

#-----------------------------STREAMLIT IMPLEMETATION------------------------------
st.header(' Project Background Information', divider = True)
st.write("The objective of this model is to create a chatbot that accurately interprets users' emotions from text inputs, providing descriptive labels for various emotional states. Using NLP and emotion recognition, it aims to enhance interaction by offering empathetic responses. Through iterative development, it seeks to improve understanding of diverse emotional expressions, contributing to mental health support, customer service, and educational applications.")

st.markdown("<h1 style = 'color: #FFFFFF; text-align: center; font-family: geneva'>EMOTION LABEL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFFFFF; text-align: center; font-family: cursive '>Built By Faith</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

user_hist =[]
reply_hist = []

robot_image, space1,space2, chats,  = st.columns(4)
with robot_image:
     robot_image.image('IMG_8499.JPG', width = 500)


with chats:
    user_message = chats.text_input('Hello user, You are chatting with Mide......how do you feel today')
def responder(text):
    user_input_processed = preprocess_text(text)
    vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
    similarity_score = cosine_similarity(vectorized_user_input, corpus)
    argument_maximum = similarity_score.argmax()
    return (data['label'].iloc[argument_maximum])

bot_greetings = ['Hello user, You are chatting with Mide......how do you feel today',
                 'Hi user, how are you feeling emotionally',
                 'Hey, how is your mood today',
                 'Hiyya, what is your current emotional state',
                 'Wassap, talk to me']

bot_farewell = ['Thank you,Have a great day..... bye',
                'Thank you so much for your time',
                'Okay, have a nice day',
                'Alright, stay safe']

human_greetings = ['hi', 'hello there', 'hey', 'hello', 'wassap']

human_exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']

import random
random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)


if user_message.lower() in human_exits:
    chats.write(f"\nchatbot:{random_farewell}!")
    user_hist.append(user_message)
    reply_hist.append(random_greeting)

elif user_message.lower() in human_greetings:
    chats.write(f"\nchatbot: {random_greeting}!")
    user_hist.append(user_message)
    reply_hist.append(random_greeting)

elif user_message == '':
    chats.write('')

else:
    response = responder(user_message)
    chats.write(f"\nchatbot: {response}")
    user_hist.append(user_message)
    reply_hist.append(random_greeting)


# Clearing Chat History 
def clearHistory():
    with open('history.txt', 'w') as file:
        pass  

    with open('reply.txt', 'w') as file:
        pass

#Save the history of user texts
import csv
with open('history.txt', 'a') as file:
    for item in user_hist:
        file.write(str(item) + '\n')

# save history of bot reply 
with open ('reply.txt', 'a') as file:
    for item in reply_hist:
        file.write(str(item) + '\n')

# Import the file to display it in the frontend
with open ('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader) 

with open('reply.txt') as f:
    reader = csv.reader(f) 
    data2 = list(reader) 

data1 = pd.Series(data1)
data2 = pd.Series(data2) 

history = pd.DataFrame({'User Input': data1, 'Bot Reply' : data2})


#history = pd.Series (data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)
st.sidebar.write(data2)


if st.button('Clear Chat History'):
    clearHistory()

#primary color - #FFFFFF
#text color -#FFFFFF
#Backgrounf color-#502A5D
#Seondary background color -#CBA8D7