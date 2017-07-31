# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:43:47 2017

@author: marzipan
"""

import os 
from threading import Thread
from subprocess import call


'''
Usage: pwd(__file__)
Returns: parent directory of "f"
'''
def pwd(f):
    return os.path.dirname(os.path.realpath(f))

def yes_no(msg):
    yn = input(msg + "\n(y/N): ")
    return yn == "Y" or yn == "y"

def start_tensorboard(directory, portno=6007):
    print("Starting TensorBoard...")
    tgt = lambda: os.system('call tensorboard --port=' + str(portno) + ' --logdir=' + directory)
    t = Thread(target=tgt)
    t.start()

def start_browser(directory, portno=6007):
    print("Opening browser...")
    tgt = lambda: os.system('explorer "http://localhost:' + str(portno) + '"')
    t = Thread(target=tgt)
    t.start()

def open_tensorboard(directory, portno=6007):
    start_tensorboard(directory, portno)
    start_browser(directory, portno)