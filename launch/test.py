import pyttsx4

engine = pyttsx4.init()
engine.setProperty('rate', 150)    # Adjust the speech rate (words per minute)

text = "Hello, my name is pcue, what can i do for you?"
engine.say(text)
engine.runAndWait()
