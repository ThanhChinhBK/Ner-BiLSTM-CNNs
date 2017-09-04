import numpy as np
from model import RNN_CNNs
import pickle, sys, json
import tensorflow as tf
from numpy import random as rnd
from sklearn.metrics import f1_score

tf.flags.DEFINE_string("train_file", "train.data", "train file path")
tf.flags.DEFINE_string("dev_file", "dev.data", "dev file path")
tf.flags.DEFINE_string("test_file", "test.data", "test file path")
tf.flags.DEFINE_string("config_file", "config.json", "config file path")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("epochs", 80, "epochs")
FLAGS = tf.flags.FLAGS

def _batch_split(token_ids, sent_len, tokens_addition, chars, chars_addition,  targets, batch_size):
    shuffled_idx = rnd.permutation(data_size)
    token_ids = token_ids[shuffled_idx]
    sent_len = sent_len[shuffled_idx]
    tokens_addition = tokens_addition[shuffled_idx]
    chars = chars[shuffled_idx]
    targets = targets[shuffled_idx]
    for i in range(0, len(targets), batch_size):
        yield token_ids[i:i+batch_size], sent_len[i:i+batch_size], tokens_addition[i:i+batch_size],\
            chars[i:i+batch_size], chars_addition[i:i+batch_size], targets[i:i+batch_size]
    

def _load_data(file_name):
    """
    data = [token_ids, sent_len, tokens_addition, chars, chars_addtion,  targets]
    """
    data = pickle.load(open(file_name, "rb"))
    sent_len = []
    targets = []
    token_ids = []
    char_ids = []
    token_addition = []
    char_addition = []
    for x in data:
        token_ids.append(x[0])
        sent_len.append(x[1])
        token_addition.append(x[2])
        char_ids.append(x[3])
        char_addition.append(x[4])
        targets.append(x[5])
    token_ids = np.array(token_ids)
    sent_len = np.array(sent_len)
    token_addition = np.array(token_addition)
    temp = np.zeros((token_addition.shape[0], token_addition.shape[1], 5))
    temp[token_addition] = 1
    token_addition = temp
    char_ids =  np.array(char_ids)
    char_addition = np.array(char_addition)
    temp = np.zeros((char_addition.shape[0], char_addition.shape[1], char_addition.shape[2], 4))
    temp[char_addition] = 1
    char_addition = temp
    targets = np.array(targets)
    return token_ids, sent_len, token_addition, char_ids, char_addition, targets

def _f1(config, predicts, labels, sent_length, f1_type="micro"):
    target = np.argmax(labels, 2)
    ytrue = []
    ypred = []
    for i in range(len(target)):
        length = sent_length[i]
        ytrue = np.concatenate((ytrue, target[i][0:length]))
        ypred = np.concatenate((ypred, predicts[i][0:length]))
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    f1 = f1_score(ytrue, 
                  ypred, 
                  labels=range(0,config["num_class"] -1),
                  pos_label=None,
                  average=f1_type)
    return f1

if __name__ == "__main__":
    sys.stderr.write("load data...")
    print(1)
    train_token_ids, train_sent_len, train_token_addition,\
        train_char_ids, train_char_addition, train_target = _load_data(FLAGS.train_file)
    dev_token_ids, dev_sent_len, dev_token_addition,\
        dev_char_ids, dev_char_addition, dev_target = _load_data(FLAGS.dev_file)
    test_token_ids, test_sent_len, test_token_addition,\
        test_char_ids, test_char_addition, test_target = _load_data(FLAGS.test_file)
    print(1)
    sys.stderr.write("done.\n")
    data_size = len(train_token_ids)
    config = json.load(open(FLAGS.config_file))
    ner = RNN_CNNs(config)
    f1_s = open("f1.txt", "w")
    for e in range(FLAGS.epochs):
        for step, (token_ids_batch, sent_len_batch, token_addition_batch,\
               char_ids_batch, char_addition_batch, target_batch) in enumerate(
                _batch_split(train_token_ids, train_sent_len, train_token_addition, train_char_ids,
                             train_char_addition, train_target, FLAGS.batch_size)):
            loss = ner.partial_fit(token_ids_batch, char_ids_batch, token_addition_batch, char_addition_batch,
                        sent_len_batch, target_batch)
            print("\repoch : {} step {} : {}".format(e, step, loss))
        prediction = ner.transform(dev_token_ids, dev_char_ids, dev_token_addition,
                                   dev_char_addition, dev_sent_len)
        f1 = _f1(config, prediction, dev_target, dev_sent_len, "micro")
        print("\nEvaluate:\n")
        print("f1 score after {} epoch:{}\n".format(e, f1))
        f1_s.write(str(f1) + "\n")
    f1_s.close()
