# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:21:22 2019

@author: Batuhan
"""

import speech_recognition as SR     # import the library

r = SR.Recognizer()
audio = SR.AudioFile("C:/Users/CEM/Downloads/k.wav")

with audio as source:
    audioData=r.record(source,offset=2,duration=4)

text = r.recognize_google(audioData,language="tr-TR")
mic = SR.Microphone()
mic.list_microphone_names()

with mic as source:
    audio = r.listen(source)

ISaid=r.recognize_google(audio)

print("You said: "+ISaid)