import sys, string
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

def clear_target(target):
    return target[2:] if len(target) > 2 else target

def clear_char(token, max_len, vocab_char):
    chars = [vocab_char[t.lower()] if t.lower() in vocab_char \
                  else vocab_char["<PAD>"] for t in token]
    chars += [0 for _ in range(max_len - len(chars))]
    return chars

def clear_char_addition(token, maxlen):
    chars_addtion = [check_addition_char(x) for x in token] 
    chars_addtion += [0 for _ in range(maxlen - len(chars_addtion))]
    return chars_addtion


