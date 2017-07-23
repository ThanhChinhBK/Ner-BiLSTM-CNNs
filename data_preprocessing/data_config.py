import json
import tensorflow as tf
import os, sys

tf.flags.DEFINE_string("dest", "../work/", "destination dir")
tf.flags.DEFINE_string("clean_dir", "../work/", "data clean dir")
FLAGS = tf.flags.FLAGS

def main():
    sents = []
    words_len = []
    config_txt = os.path.join(FLAGS.dest, 'config.txt')
    list_file = os.listdir(FLAGS.clean_dir)
    for file in list_file:
        if not file.endswith(".clean"):
            continue
        with open(FLAGS.clean_dir + file) as fi:
            for line in fi:
                tokens = json.loads(line)
                sents.append(len(tokens))
                words_len.append(max([len(token["raw"]) for token in tokens]))
    max_sent_len = max(sents)
    max_word_len = max(words_len)
    fo = open(config_txt, "w")
    fo.write("max sentence length:" + str(max_sent_len) + "\n")
    fo.write("max word length:" + str(max_word_len) + "\n")

if __name__ == "__main__":
    main()
