# from spellchecker import SpellChecker
# from nltk.metrics import edit_distance
from symspellpy.symspellpy import SymSpell, Verbosity

import os
import time
import re


# NLTK corpus
# import nltk
# nltk.download('words')
# from nltk.corpus import words
# print(word in words.words())

class Preprocessing(object):
    
    def __init__(self):
        
        self.max_dist = 2
        self.max_len = 0
        self.allowed_symbols = ".,:;?!()'"
        self.r_espression = {'numbers': re.compile(r'[+-]?(\d*\.\d+|\d+)'),
                            'symbols': re.compile(r'[\-_"#*\[\]~]'),
                            'whitespaces': re.compile(' +'),
                            }

        # self.spell = SpellChecker()
        
        # Params: initial_capacity, max_edit_distance_dictionary, prefix_length
        self.sym_spell = SymSpell(83000, self.max_dist, 7)
        dictionary_path = os.path.join(os.path.dirname(__file__),
                                       "dict/frequency_dictionary_en_82_765.txt")
        # Params: column of the term and of the term frequency in the dictionary text file
        if not self.sym_spell.load_dictionary(dictionary_path, 0, 1):
            print("Dictionary file not found")
            raise FileNotFoundError
        
        return
    
    
    # Correct wrong words
    def correct_word(self, word):

        # if word[0] == '@' or not len(self.spell.unknown([word])):
        # suggestion = self.spell.correction(word)
        
        if word[0] == '@'or word in self.allowed_symbols:
            return word

        suggestion_list = self.sym_spell.lookup(word, Verbosity.TOP, 2)

        if len(suggestion_list):
            suggestion = suggestion_list[0].term
            distance = suggestion_list[0].distance
            if distance == 0:
                return word
            else:
                return "_" + suggestion
        else:
            return "@UNKNOWN"
        
    
    # Remove symbols, extra spaces and apply word correction
    def preprocess_essay(self, essay):
        essay = self.r_espression['symbols'].sub("", essay)  # Substitute symbols in essay with spaces
        essay = self.r_espression['numbers'].sub("", essay)  # Remove numbers
        essay = essay.replace("/", " ")
        essay = essay.replace("?", " ? ")  # Isolate relevant symbols are independent tokens
        essay = essay.replace("!", " ! ")
        essay = essay.replace(".", " . ")
        essay = essay.replace(",", " , ")
        essay = essay.replace(":", " : ")
        essay = essay.replace(";", " ; ")
        essay = essay.replace("(", " ( ")
        essay = essay.replace(")", " ) ")
        essay = self.r_espression['whitespaces'].sub(" ", essay)  # Remove multiple spaces
        essay = essay.lower()
        essay = essay.split()
        
        preprocessed = []
        for w in essay:
            w = self.correct_word(w)
            # print(w)
            preprocessed.append(w)
        
        self.max_len = max(self.max_len, len(preprocessed))
        return preprocessed


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    


# spell = Preprocessing()
# # essay = " ".join(
# #     ['something', 'is', 'hapenning', 'here', 'happened', 'google', 'google', 'california', 'helfalsosfe', 'Carl', 'CAR',
# #      'Lead'])
# time_a = time.time()
# essay = spell.correct_word("memebers")
# print(time.time() - time_a)
# # print(essay)


# Params: initial_capacity, max_edit_distance_dictionary, prefix_length
# sym_spell = SymSpell(83000, 2, 7)
# dictionary_path = os.path.join(os.path.dirname(__file__),
#                                "frequency_dictionary_en_82_765.txt")
# # Params: column of the term and of the term frequency in the dictionary text file
# if not sym_spell.load_dictionary(dictionary_path, 0, 1):
#     print("Dictionary file not found")
#     raise FileNotFoundError
#
# time_a = time.time()
# suggestion = sym_spell.lookup("hebyllo", Verbosity.TOP, 2)
# print(suggestion[0].term)
# print(suggestion[0].distance)
#
# print(time.time() - time_a)