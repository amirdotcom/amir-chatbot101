import nltk 
# nltk.download('punkt', quiet=True)
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# define method for stemming
def stem(word):
    return stemmer.stem(word.lower())

# define bag or words(apply token session first, with all words)
def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hell0", "how", "are", "you"]
    words = ["hi", "hello", "I", "you","bye","thank","cool"]
    bag = [0, 1, 0, 1, 0, 0, 0, 0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words): #idx = index, w = words
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag


# to test if the array is working
# sentence = ["hell0", "how", "are", "you"]
# words = ["hi", "hello", "I", "you","bye","thank","cool"]
# bag = bag_of_words(sentence, words)
# print(bag)


# This is just to test your function for tokenize (convert the words into an array)
# a = "Hi there, what can I do for you?"
# print(a)
# tokenized_a = tokenize(a)
# print(tokenized_a)

#This is just to test your function for stem (convert the words into root words)
# words = ["ogranize", "organizes","organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

