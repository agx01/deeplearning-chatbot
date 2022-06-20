# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 03:04:53 2022

@author: Arijit Ganguly
"""

#Required imports
import numpy as np
import tensorflow as tf
import re
import time

class Chatbot:
    
    def __init__(self):
        self.main()
    
    def get_raw_data(self, file_name, encoding):
        file_dir = "../data/raw/"
        data = ""
        with open(file_dir + file_name, encoding=encoding, errors='ignore') as file:
            data = file.read().split('\n')
        return data
    
    def main(self):
        
        #Get the data of lines and conversations
        lines = self.get_raw_data("movie_lines.txt", "utf-8")
        conversations = self.get_raw_data("movie_conversations.txt", "utf-8")
        
        #Create a dict to map line numbers with lines
        id2line = {}
        
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
        
        #Create a list of all the conversations
        conversations_ids = []
        for conversation in conversations[:-1]:
            _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
            conversations_ids.append(_conversation.split(','))
            
            
        #Get the questions and answers seperately
        questions = []
        answers = []
        for conversation in conversations_ids:
            for i in range(len(conversation) - 1):
                questions.append(id2line[conversation[i]])
                answers.append(id2line[conversation[i+1]])
        
        #Debug Line
        pass
        
if __name__ == "__main__":
    bot = Chatbot()