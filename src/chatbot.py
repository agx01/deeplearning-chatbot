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
        self.id2line = {}
        self.clean_questions = []
        self.clean_answers = []
        self.word2count = {}
        self.questionwords2int = {}
        self.answerwords2int = {}
        self.questions_to_int = []
        self.answers_to_int = []
        self.sorted_clean_questions = []
        self.sorted_clean_answers = []
        self.main()
    
    def get_raw_data(self, file_name, encoding):
        file_dir = "../data/raw/"
        data = ""
        with open(file_dir + file_name, encoding=encoding, errors='ignore') as file:
            data = file.read().split('\n')
        return data
    
    def  clean_text(self, text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
        return text
    
    def preprocessing_data(self):
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
        
        #Clean questions
        clean_questions = []
        
        for question in questions:
            clean_questions.append(self.clean_text(question))
        
        
        #Clean answers
        clean_answers = []
        
        for answer in answers:
            clean_answers.append(self.clean_text(answer))
        
        #Get the word frequency for each word
        word2count = {}
        for question in clean_questions:
            for word in question.split():
                if word not in word2count:
                    word2count[word] = 1
                else:
                    word2count[word] += 1
                    
        for answer in clean_answers:
             for word in answer.split():
                 if word not in word2count:
                    word2count[word] = 1
                 else:
                    word2count[word] += 1 
        
        #Creating dictionaries for question words and answer words to a
        #unique int value
        threshold = 20
        questionwords2int = {}
        word_number = 0
        
        for word, count in word2count.items():
            if count >= threshold:
                questionwords2int[word] = word_number
                word_number += 1
        
        answerwords2int = {}
        word_number = 0
        
        for word, count in word2count.items():
            if count >= threshold:
                answerwords2int[word] = word_number
                word_number += 1
    
        #Add the last tokens to these 2 dictionaries
        tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
        
        for token in tokens:
            questionwords2int[token] = len(questionwords2int) + 1
        
        for token in tokens:
            answerwords2int[token] = len(answerwords2int) + 1
    
        #Create inverse dictionary of the answerword2int dictionary
        answersints2word = {w_i: w for w, w_i in answerwords2int.items()}
        
        #Adding the EOS string on each of the answers
        for i in range(len(clean_answers)):
            clean_answers[i] += " <EOS>"
        
        #Translating all the questions and the answers into integers
        #and replacing all the words that were filtered out by <OUT>
        questions_to_int = []
        for question in clean_questions:
            ints = []
            for word in question.split():
                if word not in questionwords2int:
                    ints.append(questionwords2int['<OUT>'])
                else:
                    ints.append(questionwords2int[word])
            questions_to_int.append(ints)
        
        answers_to_int = []
        for answer in clean_answers:
            ints = []
            for word in answer.split():
                if word not in answerwords2int:
                    ints.append(answerwords2int['<OUT>'])
                else:
                    ints.append(answerwords2int[word])
            answers_to_int.append(ints)
        
    
        #Sort the question and answers by the length of questions
        sorted_clean_questions = []
        sorted_clean_answers = []
        
        for length in range(1, 25 + 1):
            for i in enumerate(questions_to_int):
                if len(i[1]) == length:
                    sorted_clean_questions.append(questions_to_int[i[0]])
                    sorted_clean_answers.append(answers_to_int[i[0]])
                    
        #Assigning to class attributes
        self.id2line = id2line
        self.clean_questions = clean_questions
        self.clean_answers = clean_answers
        self.word2count = word2count
        self.questionwords2int = questionwords2int
        self.answerwords2int = answerwords2int
        self.questions_to_int = questions_to_int
        self.answers_to_int = answers_to_int
        self.sorted_clean_questions = sorted_clean_questions
        self.sorted_clean_answers = sorted_clean_answers
        
    def model_inputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='target')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, targets, lr, keep_prob
    
    def preprocess_targets(self, targets, word2int, batch_size):
        left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
        right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1])
        preprocessed_targets = tf.concat([left_side, right_side], 1)
        return preprocessed_targets
    
    def main(self):
        self.preprocessing_data()
        
    
        
if __name__ == "__main__":
    bot = Chatbot()