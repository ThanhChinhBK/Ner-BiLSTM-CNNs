import numpy as np
from model import RNN_CNNs
import pickle, sys, json, utils
import tensorflow as tf
from numpy import random as rnd
from sklearn.metrics import f1_score
import data_encode as de 

tf.flags.DEFINE_string("train_file", "data/train", "train file path")
tf.flags.DEFINE_string("dev_file", "data/dev", "dev file path")
tf.flags.DEFINE_string("test_file", "data/test", "test file path")
tf.flags.DEFINE_string("config_file", "config.json", "config file path")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("epochs", 80, "epochs")
tf.flags.DEFINE_string("embedding_path", "glove.6B.50d.txt", "word embedding path")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("PadZeroBegin", False, "where to pad zero in the input")

FLAGS = tf.flags.FLAGS
logger = utils.get_logger("MainCode")

def _batch_split(token_ids, sent_len, chars, targets, batch_size):
    data_size = len(token_ids)
    shuffled_idx = rnd.permutation(data_size)
    token_ids = np.array(token_ids)[shuffled_idx]
    sent_len = np.array(sent_len)[shuffled_idx]
    chars = np.array(chars)[shuffled_idx]
    targets = np.array(targets)[shuffled_idx]
    for i in range(0, len(targets), batch_size):
        yield token_ids[i:i+batch_size], sent_len[i:i+batch_size],\
            chars[i:i+batch_size],targets[i:i+batch_size]
    



def _f1(config, predicts, labels, sent_length, f1_type="micro"):
    #target = np.argmax(labels, 2)
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
                  labels=range(0,config["num_class"]),
                  pos_label=None,
                  average=f1_type)
    return f1

if __name__ == "__main__":
    config = json.load(open(FLAGS.config_file))
    logger.info("load data...")
    word_alphabet = ["<None>"]
    label_alphabet = ['O', "PER", "MISC", "ORG", "LOC"]
    word_sentences_train, label_sentences_train, word_id_sentences_train, label_id_sentences_train \
        = de.read_conll_sequence_labeling(FLAGS.train_file, word_alphabet, label_alphabet)
    word_sentences_dev, label_sentences_dev, word_id_sentences_dev, label_id_sentences_dev \
        = de.read_conll_sequence_labeling(FLAGS.dev_file, word_alphabet, label_alphabet)
    word_sentences_test, label_sentences_test, word_id_sentences_test, label_id_sentences_test \
        = de.read_conll_sequence_labeling(FLAGS.test_file, word_alphabet, label_alphabet)
    max_length_train = utils.get_max_length(word_id_sentences_train)
    max_length_dev = utils.get_max_length(word_id_sentences_dev)
    max_length_test = utils.get_max_length(word_id_sentences_test)
    max_length = max(max_length_train, max_length_dev, max_length_test)
    logger.info("done.\n")
    config["sentence_length"] = max_length
    logger.info("word alphabet size: %d" % (len(word_alphabet) - 1))
    logger.info("label alphabet size: %d" % (len(label_alphabet) - 1))
    logger.info("set max sentence length to %d" %(max_length))
    logger.info("Padding training text and lables ...")
    char_alphabet = ["<None>"]
    word_index_sentences_train_pad,train_seq_length = utils.padSequence(word_id_sentences_train,
                                                                        max_length, 
                                                                        beginZero=FLAGS.PadZeroBegin)
    label_index_sentences_train_pad,_= utils.padSequence(label_id_sentences_train,
                                                         max_length, 
                                                         beginZero=FLAGS.PadZeroBegin)

    logger.info("Padding dev text and labels ...")
    word_index_sentences_dev_pad,dev_seq_length = utils.padSequence(word_id_sentences_dev,
                                                                    max_length, 
                                                                    beginZero=FLAGS.PadZeroBegin)
    label_index_sentences_dev_pad,_= utils.padSequence(label_id_sentences_dev,
                                                       max_length, 
                                                       beginZero=FLAGS.PadZeroBegin)

    char_index_train,max_char_per_word_train= de.generate_character_data(word_sentences_train,  
                                                                         char_alphabet,
                                                                         setType="Train")
    logger.info("Creating character set FROM dev set ...")
    char_index_dev,max_char_per_word_dev= de.generate_character_data(word_sentences_dev, 
                                                                     char_alphabet, 
                                                                     setType="Dev",
                                                                     train_abble=False)
    logger.info("character alphabet size: %d" % (len(char_alphabet) - 1))
    max_char_per_word = min(de.MAX_CHAR_PER_WORD, max_char_per_word_train,max_char_per_word_dev)
    config["word_length"] = max_char_per_word    
    logger.info("set Maximum character length to %d" %max_char_per_word)
    logger.info("Padding Training set ...")
    char_index_train_pad = de.construct_padded_char(char_index_train, char_alphabet, 
                                                    max_sent_length=max_length,max_char_per_word=max_char_per_word)
    logger.info("Padding Dev set ...")
    char_index_dev_pad = de.construct_padded_char(char_index_dev, char_alphabet, 
                                                  max_sent_length=max_length,max_char_per_word=max_char_per_word)

    embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict("glove", 
                                                                       FLAGS.embedding_path,
                                                                       logger)
    logger.info("Dimension of embedding is %d, Caseless: %d" % (embedd_dim, caseless))
    embedd_table = de.build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless)
    char_embedd_table = de.build_char_embedd_table(char_alphabet, config["char_embded_size"])
    logger.info("build embedding complete")
    ner = RNN_CNNs(config, embedd_table, char_embedd_table)
    logger.info("Model Created")
    f1_s = open("f1.txt", "w")
    for e in range(FLAGS.epochs):
        for step, (token_ids_batch, sent_len_batch,\
               char_ids_batch, target_batch) in enumerate(
                _batch_split( word_index_sentences_train_pad, train_seq_length, char_index_train_pad,
                              label_index_sentences_train_pad, FLAGS.batch_size)):
            loss = ner.partial_fit(token_ids_batch, char_ids_batch, 
                                   sent_len_batch, target_batch)
            print("\repoch : {} step {} : {}".format(e, step, loss))
        dev_prediction = ner.transform(word_index_sentences_dev_pad, char_index_dev_pad, dev_seq_length)
        f1 = _f1(config, dev_prediction, label_index_sentences_dev_pad, dev_sent_len, "micro")
        print("\nEvaluate:\n")
        print("f1 score after {} epoch:{}\n".format(e, f1))
        f1_s.write(str(f1) + "\n")
    f1_s.close()
