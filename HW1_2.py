from codecs import ignore_errors
import pandas as pd
import numpy as np
import os
import sys
import re
import warnings
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

warnings.filterwarnings('ignore')


# Funtion to read the corpus and change all alphabets to lower case
def read(file):
    f = open(file,'r')
    lines = f.readlines()               
    for i in range(len(lines)):
        lines[i] = re.findall(r'[^\s][\w]+[^\s]',lines[i].lower())   
    return lines



def pad(top, left,lines):
    top_s = len(top)    
    left_s = len(left) 
    padding = np.zeros((left_s, top_s), dtype = 'double')
    for word in lines:
        for l in range(left_s):
            first_word = left[l]
            if first_word in word:
                for t in range(top_s):
                    second_word = top[t]
                    first_index = word.index(first_word)
                    if second_word in word:
                        second_index = word.index(second_word)
                        if second_index >= (first_index - 5) and second_index <= (first_index + 5):  
                            padding[l][t] += 1

    num = sum(sum(padding))
    padding /= num
    return padding


# Function to calculate the PPMI values of top-left words
def ppmi(top, left, vec):
    left_v = vec.sum(axis = 1)     
    top_v = vec.sum(axis = 0)      
    top_s = len(top)
    left_s = len(left)
    vec_ppmi = np. zeros((left_s, top_s))
    for l in range(left_s):
        for t in range(top_s):
            vec_ppmi[l][t] = round(np.fmax(((np.log2(vec[l][t] / (left_v[l] * top_v[t])))), 0),3) 
            if (left[l] == "chairman" and top[t] == "said") or (left[l] == "chairman" and top[t] == "of") or (left[l] == "company" and top[t] == "said") or (left[l] == "company" and top[t] == "board"):
                print("The word ", left[l], " for the context-word ", top[t],": ", vec_ppmi[l][t])

    return vec_ppmi


# Function to calculate the similarity vector
def calculate_similarity(vec):
    dim = len(vec)
    similarity = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            similarity[i][j] = np.dot(vec[i],vec[j])/(np.sqrt(np.dot(vec[i],vec[i]))*np.sqrt(np.dot(vec[j],vec[j]))) 
    return  similarity


def main():
    corpus = "corpus_for_language_models.txt"
    lines = read(corpus)
    words_top = ['said', 'of', 'board']         
    words_left = ['chairman', 'company']       

    print("For Problem 2-1") 
    padding = pad(words_top, words_left,lines)
    ppmi(words_top, words_left, padding)

    words_top = ['said', 'of', 'board']                             
    words_left = ['chairman', 'company', 'sales', 'economy']        
    left_s = len(words_left)

    print("\nFor Problem 2-2") 
    word_vec = pad(words_top, words_left,lines)
    similarity = calculate_similarity(word_vec)
    print()
    for i in range(left_s):
        for j in range(i + 1, left_s):
            if (words_left[i] == 'chairman' and words_left[j] == 'company') or (words_left[i] == 'company' and words_left[j] == 'sales') or (words_left[i] == 'company' and words_left[j] == 'economy'):
                print("Similarity among: ", words_left[i], words_left[j], " = ", similarity[i][j])
    print("The words: chairman and company are most similar among the given list i.e. the occurances of these words with the given context words in the corpus has very similar meanings and associations.")
    print("Although, the values are not perfectly 1 as the word company is associated with the word chairman many times but the word chairman be associated with other words or has different meaning in the given corpus")
main()
