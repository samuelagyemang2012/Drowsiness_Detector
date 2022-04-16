import pyttsx3

engine = pyttsx3.init()


# read warning
def warning_voice(text):
    engine.say(text)
    engine.runAndWait()
