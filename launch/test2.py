import sys
import threading
import tkinter as tk

import pyttsx4 as tts
import speech_recognition as sr
from neuralintents import GenericAssistant

class P_CUE:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.speaker = tts.init()
        self.speaker.setProperty("rate",150)
        
        self.assistant = GenericAssistant("launch/intents.json",intent_methods={"file":self.create_file})
        self.assistant.train_model()
        
        self.root = tk.Tk()
        self.label = tk.Label(text="🤖", font = ("Arial",120,"bold"))
        self.label.pack()
        
        threading.Thread(target=self.run_pcue).start()
        
        self.root.mainloop()
        
    def create_file(self):
        with open("somefile.txt", "w") as f:
            f.write("hello test word")
        
    def run_pcue(self):
        while True:
            try:
                with sr.Microphone() as mic:
                    self.recognizer.adjust_for_ambient_noise(mic,duration = 0.2)
                    print("Am listening....")
                    audio = self.recognizer.listen(mic)
                    
                    text = self.recognizer.recognize_google(audio)
                    text = text.lower()
                    
                    if "hey prime" in text:
                        self.label.config(fg="red")
                        print("Am listening....")
                        audio = self.recognizer.listen(mic)
                        text = self.recognizer.recognize_google(audio)
                        text = text.lower()
                        if text == "stop":
                            self.speaker.say("Ok GoodBye!")
                            self.speaker.runAndWait()
                            self.speaker.stop()
                            self.root.destroy()
                            sys.exit()
                        else:
                            if text is not None:
                                response = self.assistant.request(text)
                                if response is None:
                                    self.speaker.say(response)
                                    self.speaker.runAndWait()
                            self.label.config(fg='black')
            except:
                 self.label.config(fg='black')
                 continue
             
P_CUE()