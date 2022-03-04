import pandas as pd
import numpy as np
import os
import sys
import re

# Given sentences
s1 = "Sales of the company to return to normalcy."
s2 = "The new products and services contributed to increase revenue."


# Function to generate N-Grams
def get_NGrams(data, n):
    words = [trimSpecialChar(word) for word in data.split(" ")]  
    temp = zip(*[words[i:] for i in range(0,n)])
    ans = [' '.join(n) for n in temp]
    return ans


# Funtion to remove all special characters from a string
def trimSpecialChar(data):
    word = ''.join(char for char in data if char.isalnum()).lower()
    return word


# Funtion to read the given corpus
def read():
    file = os.path.join("corpus_for_language_models.txt")
    global tokenizedData
    tokenizedData = []

    with open(file) as f:
        data = f.read().split()
        for d in data:
            temp = trimSpecialChar(d)
            if temp != '':
                tokenizedData.append(temp)
    global vocab
    vocab = set(tokenizedData)
    global trigramData
    trigramData = get_NGrams(" ".join(tokenizedData), 3)
    global bigramData
    bigramData = get_NGrams(" ".join(tokenizedData), 2)
    global unigramData
    unigramData = get_NGrams(" ".join(tokenizedData), 1)


# 1. Write a program to compute the trigrams for any given input. TOTAL: 2 points
#    Apply your program to compute the trigrams you need for sentences S1 and S2.
def sol1A1():
    print("\nSolution for 1A -1")
    global s1Trigram
    s1Trigram = get_NGrams(s1, 3)
    global s2Trigram
    s2Trigram = get_NGrams(s2, 3)
    print("Trigram for S1: ", s1Trigram)
    print("Trigram for S2: ", s2Trigram)


# Function to get the count for n-grams
def get_ngramCount(tokens, n, k=0):
    temp = []
    current = []
    if n == 3:
        current = trigramData
    elif n == 2:
        current = bigramData
    elif n == 1:
        current = unigramData
    for token in tokens:
        tg_count = current.count(token) + k
        temp.append(tg_count)
    return temp


# Function to get the probability for n-grams
def get_ngramProb(tokens, n, k=0):
    temp = []
    current = []
    prev = []
    if n == 3:
        current = trigramData
        prev = bigramData
    elif n == 2:
        current = bigramData
        prev = unigramData
    elif n == 1:
        current = unigramData
    for token in tokens:
        tg_count = current.count(token) + k
        if n == 1:
            bg_count = len(tokenizedData)
        else:
            bg_count = prev.count(" ".join(token.split(" ")[:n-1])) + (0 if k == 0 else k * len(vocab))
        temp.append(tg_count/bg_count)
    return temp


#2. Construct automatically (by the program) the tables with 
# (a) the trigram counts (2 points) and 
# (b) trigram probabilities for the language model without smoothing. (3 points)
def sol1A2():
    print("\nSolution for 1A -2")
    global s1_trigramdf
    s1_trigramdf = pd.DataFrame()
    global s2_trigramdf
    s2_trigramdf = pd.DataFrame()
    s1_trigramdf["grams"] = s1Trigram
    s2_trigramdf["grams"] = s2Trigram
    s1_trigramdf["noSmoothing_count"] = get_ngramCount(s1Trigram, 3)
    s1_trigramdf["noSmoothing_prob"] = get_ngramProb(s1Trigram, 3)
    s2_trigramdf["noSmoothing_count"] = get_ngramCount(s2Trigram, 3)
    s2_trigramdf["noSmoothing_prob"] =  get_ngramProb(s2Trigram, 3)
    print("Trigram counts: ")
    print("S1: \n",s1_trigramdf[["grams" , "noSmoothing_count"]])
    print("S2: \n",s2_trigramdf[["grams" , "noSmoothing_count"]])

    print("Trigram probabilities without smoothing")
    print("S1: \n",s1_trigramdf[["grams" , "noSmoothing_prob"]])
    print("S2: \n",s2_trigramdf[["grams" , "noSmoothing_prob"]])


# Funtion to calculate reconstituted counts
def get_reconstitutedCounts(trigram, prob):
    return prob * bigramData.count(" ".join(trigram.split(" ")[:2]))

# 3. Construct automatically (by the program): (i) the Laplace-smoothed count tables; (2
#   points) (ii) the Laplace-smoothed probability tables (3 points); and (iii) the
#   corresponding re-constituted counts (3 points)
def sol1A3():
    print("\nSolution for 1A -3")
    global s1_trigramdf, s2_trigramdf
    s1_trigramdf["Smoothing_count"], s1_trigramdf["Smoothing_prob"] = get_ngramCount(s1Trigram, 3,  1), get_ngramProb(s1Trigram, 3, 1)
    s2_trigramdf["Smoothing_count"], s2_trigramdf["Smoothing_prob"] = get_ngramCount(s2Trigram, 3, 1), get_ngramProb(s2Trigram, 3, 1)
    
    print("Laplace smoothed count tables: ")
    print("S1: \n",s1_trigramdf[["grams" , "Smoothing_count"]])
    print("S2: \n",s2_trigramdf[["grams" , "Smoothing_count"]])

    print("Laplace smoothed probability tables: ")
    print("S1: \n",s1_trigramdf[["grams" , "Smoothing_prob"]])
    print("S2: \n",s2_trigramdf[["grams" , "Smoothing_prob"]])   

    print("Corresponding re-constituted counts: ")
    s1_trigramdf["reconstituted_count"] = [get_reconstitutedCounts(trigram, prob) for trigram, prob in zip(s1_trigramdf["grams"], s1_trigramdf["Smoothing_prob"])]
    s2_trigramdf["reconstituted_count"] = [get_reconstitutedCounts(trigram, prob) for trigram, prob in zip(s2_trigramdf["grams"], s2_trigramdf["Smoothing_prob"])]

    print("S1: \n", s1_trigramdf[["grams","reconstituted_count"]])
    print("S2: \n", s2_trigramdf[["grams","reconstituted_count"]])


# 4. Construct automatically (by the program) the smoothed trigram probabilities using the Katz back-off method. (8 points) 
#   How many times you had to also compute the smoothed trigram probabilities (2 points) 
#   How many times you had to compute the smoothed unigram probabilities (2 points). 
def sol1A4():

    # Function to calculate the Probability Count Table
    def get_probCountTable(s, ngram):
        ngramss = get_NGrams(s, ngram)
        cp = pd.DataFrame()
        cp["grams"] = ngramss
        cp["noSmoothing_count"], cp["noSmoothing_prob"] = get_ngramCount(ngramss, ngram), get_ngramProb(ngramss, ngram)
        cp["Smoothing_count"], cp["Smoothing_prob"] = get_ngramCount(ngramss, ngram), get_ngramProb(ngramss, ngram)
        cp["reconstituted_count"] = [get_reconstitutedCounts(trigram, prob) for trigram, prob in zip(cp["grams"], cp["Smoothing_prob"])]
        cp["discounted_prob"] = cp["reconstituted_count"] / cp["noSmoothing_count"]
        cp["discounted_prob"].replace(np.inf, 0, inplace=True)
        cp["discounted_prob"].replace(np.nan, 0, inplace=True)
        return cp

    print("\nSolution for 1A -4")
    global s1_trigramdf, s2_trigramdf, s1_bigram_cp, s1_unigram_cp, s2_bigramdf, s2_unigramdf
    global s1_trigramProb_counts, s1_bigramProb_counts, s1_unigram_probCount, s2_trigram_probCount, s2_bigram_probCount, s2_unigram_probCount
    s1_trigramProb_counts, s1_bigramProb_counts, s1_unigram_probCount, s2_trigram_probCount, s2_bigram_probCount, s2_unigram_probCount = 0,0,0,0,0,0
    s1_trigramdf["discounted_count"] = s1_trigramdf["reconstituted_count"] / s1_trigramdf["noSmoothing_count"]
    s1_trigramdf["discounted_count"].replace(np.inf, 0, inplace=True)
    print(s1_trigramdf[["grams", "discounted_count"]])

    s1_trigramdf["discounted_prob"] = s1_trigramdf["reconstituted_count"] / s1_trigramdf["noSmoothing_count"]
    s1_trigramdf["discounted_prob"].replace(np.inf, 0, inplace=True)
    s1_trigramdf["discounted_prob"].replace(np.nan, 0, inplace=True)
    
    s1_bigram_cp = get_probCountTable(s1, 2)

    s1_unigram_cp = get_probCountTable(s1, 1)

    temp = [get_katzProb(trigram, c, dp, 3, s1_trigramdf[s1_trigramdf["grams"] == trigram], s1_bigram_cp, s1_unigram_cp, True) for trigram, c, dp in zip(s1_trigramdf["grams"], s1_trigramdf["noSmoothing_count"], s1_trigramdf["discounted_count"])]
    s1_trigramdf["katzProb"] = temp
    print("For S1: \n",s1_trigramdf[["grams", "katzProb"]])
    print("Number of times smoothed trigram probabilities were computed: ",s1_trigramProb_counts)
    print("Number of times smoothed trigram probabilities were computed: ", s1_unigram_probCount)

    s2_trigramdf["discounted_count"] = s2_trigramdf["reconstituted_count"] / s2_trigramdf["noSmoothing_count"]
    s2_trigramdf["discounted_count"].replace(np.inf, 0, inplace=True)
    print(s2_trigramdf[["grams", "discounted_count"]])


    s2_trigramdf["discounted_prob"] = s2_trigramdf["reconstituted_count"] / s2_trigramdf["noSmoothing_count"]
    s2_trigramdf["discounted_prob"].replace(np.inf, 0, inplace=True)
    s2_trigramdf["discounted_prob"].replace(np.nan, 0, inplace=True)

    s2_bigramdf = get_probCountTable(s2, 2)
    s2_unigramdf = get_probCountTable(s2, 1)

    temp = [get_katzProb(trigram, c, dp, 3, s2_trigramdf[s2_trigramdf["grams"] == trigram], s2_bigramdf, s2_unigramdf, False) for trigram, c, dp in zip(s2_trigramdf["grams"], s2_trigramdf["noSmoothing_count"], s2_trigramdf["discounted_count"])]
    s2_trigramdf["katzProb"] = temp

    print("For S2: \n",s2_trigramdf[["grams", "katzProb"]])
    print("Number of times smoothed trigram probabilities were computed: ",s2_trigram_probCount)
    print("Number of times smoothed trigram probabilities were computed: ", s2_unigram_probCount)


# Function to calculate the Katz Probability
def get_katzProb(token, c, dp, ngram, df, bg, ug, is_s1):

    def get_alpha(dp_n, dp_d):
        if len(dp_d) == 1 and dp_d.iloc[0] == -1:
            return (1 - dp_n.sum())
        return (1 - dp_n.sum())/(1-dp_d.sum())
        
    global s1_trigramProb_counts, s1_bigramProb_counts, s1_unigram_probCount, s2_trigram_probCount, s2_bigram_probCount, s2_unigram_probCount
    token = " ".join(token.split(" ")[:ngram-1])
    if c > 0:
        if is_s1:
            s1_trigramProb_counts += 1
        else:
            s2_trigram_probCount += 1
        return dp
    elif ngram == 3:
        if is_s1:
            s1_bigramProb_counts += 1
        else:
            s2_bigram_probCount += 1
        tempp = bg[bg["grams"] == token]
        new_c = 0 if tempp["noSmoothing_count"].empty else tempp["noSmoothing_count"].iloc[0]
        new_dp = 0 if tempp["discounted_prob"].empty else tempp["discounted_prob"].iloc[0]
        dp_n = df[df["noSmoothing_count"] > 0]["discounted_prob"]
        dp_d = tempp[tempp["noSmoothing_count"] > 0]["discounted_prob"]
        return get_alpha(dp_n, dp_d) * get_katzProb(token, new_c, new_dp, ngram - 1, tempp, bg, ug, is_s1)
    elif ngram == 2:
        if is_s1:
            s1_unigram_probCount += 1
        else:
            s2_unigram_probCount += 1
        tempp = ug[ug["grams"] == token]
        new_c = 0 if tempp["noSmoothing_count"].empty else tempp["noSmoothing_count"].iloc[0]
        new_dp = 0 if tempp["discounted_prob"].empty else tempp["discounted_prob"].iloc[0]
        dp_n = df[df["noSmoothing_count"] > 0]["discounted_prob"]
        dp_d = tempp[tempp["noSmoothing_count"] > 0]["discounted_prob"]
        return get_alpha(dp_n, dp_d) * get_katzProb(token, new_c, new_dp, ngram - 1, tempp, bg, ug, is_s1)
    dp_n = df[df["noSmoothing_count"] > 0]["discounted_prob"]
    dp_d = pd.Series([-1])
    return get_alpha(dp_n, dp_d)


# 5. Compute the total probabilities for each sentence S1 and S2, when (a) using the
#   trigram model without smoothing; (1 points) and (b) when using the trigram model
#   Laplace-smoothed (1 points), as well when using the trigram probabilities resulting from
#   the Katz back-off smoothing (1 points)
def sol1A5():
    print("\nSolution for 1A -5")
    global s1_trigramdf, s2_trigramdf
    print("Total probabilities without smoothing for S1: ",s1_trigramdf[["noSmoothing_prob"]].sum().iloc[0])
    print("Total probabilities without smoothing for S2: ",s2_trigramdf[["noSmoothing_prob"]].sum().iloc[0])

    print("Total probabilities with Laplace-smoothed for S1: ",s1_trigramdf[["Smoothing_prob"]].sum().iloc[0])
    print("Total probabilities with Laplace-smoothed for S2: ",s2_trigramdf[["Smoothing_prob"]].sum().iloc[0])

    print("Total probabilities with Katz back-off smoothing for S1: ",s1_trigramdf[["katzProb"]].sum().iloc[0])
    print("Total probabilities with Katz back-off smoothing for S2: ",s2_trigramdf[["katzProb"]].sum().iloc[0])

def main():
    read()
    sol1A1()
    sol1A2()
    sol1A3()
    sol1A4()
    sol1A5()

main()

