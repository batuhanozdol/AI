# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:10:52 2019

@author: Batuhan
"""
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS as tts

def capture():
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        print("Sizi dinliyorum.")
        audio = rec.listen(source,phrase_time_limit=4)  ## 4 saniye dinleme
    try:
        text = rec.recognize_google(audio,language="tr-TR") ## dil desteği
        return text
    except:
        speak("Üzgünüm, anlayamadım .")
        return " "  


#def process_text(name,input):
 #   speak(name+" "+input+" dedi.")
  #  return 


def speak(text):
    print(text)
    speech = tts(text=text,lang='tr')
    speech_file= 'input.mp3'
    speech.save(speech_file)
    sound = AudioSegment.from_mp3(speech_file)
    play(sound)
    os.remove(speech_file)

pc = {"asus":15,"lenovo":10,"toshiba":9,"apple":10} 
   
speak('Hoşgeldiniz, isminiz nedir')
name=capture()
speak('Merhaba '+name)
speak("Nasılsın ?")
while 1:
    print("Konuşmayı bitirmek için bitti diyin.")
    captured_text = str(capture()).lower()
    if 'bitti' in str(captured_text):
        speak("Görüşmek üzere")
        break
    elif 'iyiyim' in captured_text or 'iyi' in captured_text :
        speak("Ben de iyiyim .")
        speak("Nasıl yardım edebilirim ?")
        print("Konuşmayı bitirmek için bitti diyin.")
        captured_text = str(capture()).lower()
        if 'bitti' in captured_text:
            speak("Görüşmek üzere")
            break
        else:
            if "bilgisayar" in captured_text:
                speak("İstediğiniz modeli söyleyin.")
                captured_text = str(capture()).lower()
                if pc[captured_text] != None:
                    speak("Bu modelden "+ str(pc[captured_text]) +" adet var.")
                    
            else:
                print(captured_text)
    elif 'kötüyüm' in captured_text or "kötü" in captured_text:
        speak("Kötü olmanıza üzüldüm.")
        speak("Nasıl yardım edebilirim ?")
        print("Konuşmayı bitirmek için bitti diyin.")
        captured_text = str(capture()).lower()
        if 'bitti' in captured_text:
            speak("Görüşmek üzere")
            break
        else:
            if "bilgisayar" in captured_text:
                speak("İstediğiniz modeli söyleyin.")
                captured_text = str(capture()).lower()
                if pc[captured_text] != None:
                    speak("Bu modelden "+ str(pc[captured_text]) +" adet var.")
                    print("Konuşmayı bitirmek için bitti diyin.")
                    
            else:
                print(captured_text)
  
    
    
    
    
    
    
    # process_text(name,captured_text)
    



