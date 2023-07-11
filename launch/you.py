import speech_recognition as sr

def listen_for_speech():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(audio)
        print("Speech:", text)
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print("Error:", e)

listen_for_speech()
