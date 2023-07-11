import random
import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pyttsx4 as tts
import speech_recognition as sr
#have import test2
import sys
import threading
import tkinter as tk

import pyttsx4

speaker = pyttsx4.init()
speaker.setProperty('rate',150)
recognizer = sr.Recognizer()


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('launch/intents.json').read())

words = pickle.load(open('words_1.pkl','rb'))
classes = pickle.load(open('classes_1.pkl','rb'))
model = load_model('p-cue_v1.1.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key= lambda x: x[1],reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent':classes[r[0]],'probability': str(r[1])})

    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def run_pcue():
    while True:
            try:
                with sr.Microphone() as mic:
                    recognizer.adjust_for_ambient_noise(mic,duration = 0.2)
                    audio = recognizer.listen(mic)
                    
                    text = recognizer.recognize_google(audio)
                    text = text.lower()
                    
                    if "hey prime" in text:
                        label.config(fg="red")
                        audio = recognizer.listen(mic)
                        text = recognizer.recognize_google(audio)
                        text = text.lower()
                        if text == "stop":
                            speaker.say("Ok GoodBye!")
                            speaker.runAndWait()
                            speaker.stop()
                            root.destroy()
                            sys.exit()
                        else:
                            if text is not None:
                                response = predict_class(text)
                                if response is None:
                                    speaker.say(response)
                                    speaker.runAndWait()
                            label.config(fg='black')
            except:
                 label.config(fg='black')
                 continue
             

root = tk.Tk()
label = tk.Label(text="🤖", font = ("Arial",120,"bold"))
label.pack()
threading.Thread(target=run_pcue).start()


print("Waiting for command to complete")
while True:
   
    message = input("")
    ints = predict_class(message)
    res = get_response(ints,intents)
    speaker.say(res)
    speaker.runAndWait()
    print(res)
 
