from symspellpy.symspellpy import SymSpell, Verbosity

import os
import re


class Preprocessing(object):
    
    def __init__(self, global_misspelled_token):
        
        self.global_misspelled_token = global_misspelled_token
        self.max_dist = 2
        self.max_len = 0
        self.allowed_symbols = ".,:;?!()'"
        self.r_espression = {'numbers': re.compile(r'[+-]?(\d*\.\d+|\d+)'),
                            'symbols': re.compile(r'[\-_"#*\[\]~]'),
                            'whitespaces': re.compile(' +'),
                            }

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
        
        if word[0] == '@'or word in self.allowed_symbols:
            return [word]

        suggestion_list = self.sym_spell.lookup(word, Verbosity.TOP, 2)

        if len(suggestion_list):
            suggestion = suggestion_list[0].term
            distance = suggestion_list[0].distance
            if distance == 0:
                return [word]
            else:
                if self.global_misspelled_token:
                    return ["@correction_token", suggestion]
                else:
                    return ["_" + suggestion]
        else:
            return ["@misspelled"]
        
    
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
            preprocessed += w
        
        self.max_len = max(self.max_len, len(preprocessed))

        return preprocessed


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.add_word('@padding')  # Zero padding at the end of essays
        self.add_word('@unknown')  # Word is too unfrequent or did not appear in the trainset, and set as unknown
        self.add_word('@misspelled')  # The word is misspelled, with an editing distance > 2 to any known word
        self.add_word('@correction_token')  # The next word was corrected


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

