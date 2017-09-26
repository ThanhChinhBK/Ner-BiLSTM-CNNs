import sys, string, array
import json
import numpy as np
from collections import defaultdict

def check_additon_word(token):
    '''
    Additional Word-level Features
    noinfo = 0, allCaps = 1, lowercase=2, mixedCaps=3, upperInit = 4
    '''
    if token.isupper(): return 1
    elif token.islower(): return 2
    elif token.istitle(): return 4
    else:
        check_noinfo = False
        for c in token:
            if not c.isupper() and not c.islower():
                check_noinfo = True
                break
        if check_noinfo: return 0
        else: return 3

def check_addition_char(c):
    '''
    Additional Character-level Features
    other = 0, lower = 1, upper = 2, punct = 3
    '''
    if c == '<PAD>': return 0
    if c.islower(): return 1
    elif c.isupper(): return 2
    elif c in string.punctuation: return 3
    else: return 0

def clear_target(target=""):
    target = target[2:] if len(target) > 2 else target
    return target

def clear_char(token, max_len, vocab_char):
    chars = [vocab_char[t.lower()] if t.lower() in vocab_char \
                  else vocab_char["<PAD>"] for t in token]
    chars += [0 for _ in range(max_len - len(chars))]
    return chars

def clear_char_addition(token, maxlen):
    chars_addtion = [check_addition_char(x) for x in token] 
    chars_addtion += [0 for _ in range(maxlen - len(chars_addtion))]
    return chars_addtion


def load_word_vector(vocab_path, vector_path):
     with open(vocab_path, "r") as fi:
            vocab = {x.split()[0]: int(x.split()[1]) for x in fi}
            id2vocab = {v: k for k,v in self.vocab.items()}
    
def load_glove_vectors(filename, vocab):
  """
  Load glove vectors from a .txt file.
  Optionally limit the vocabulary to save memory. `vocab` should be a set.
  """
  dct = {}
  vectors = array.array('d')
  current_idx = 0
  with open(filename, "r", encoding="utf-8") as f:
    for _, line in enumerate(f):
      tokens = line.split(" ")
      word = tokens[0]
      entries = tokens[1:]
      if not vocab or word in vocab:
        dct[word] = current_idx
        vectors.extend(float(x) for x in entries)
        current_idx += 1
    word_dim = len(entries)
    num_vectors = len(dct)
    
    return [np.array(vectors).reshape(num_vectors, word_dim), dct]

def build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vec):
    finded = 0
    vocab_vec = np.zeros((len(vocab_dict), glove_vec.shape[1]))
    for word in vocab_dict.keys():
        if word in glove_dict.keys():
            vocab_vec[vocab_dict[word]] = glove_vec[glove_dict[word]]
            finded += 1
        else:
            vocab_vec[vocab_dict[word]] = np.random.uniform(-0.25, 0.25, glove_vec.shape[1])
    sys.stderr.write("Found {} out of {} vectors in Glove...".format(finded, len(vocab_dict)))
    return vocab_vec
