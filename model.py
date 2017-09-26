import tensorflow as tf
import json, pickle
from datetime import datetime

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

class RNN_CNNs():

    def _add_placeholder(self):
        self.sentence = tf.placeholder(tf.int32, [None, self.config["sentence_length"]],
                                       name="sentence")
        self.word_list = tf.placeholder(tf.int32, [None, self.config["sentence_length"], self.config["word_length"]],
                                        name="word_list")
        #self.word_addition = tf.placeholder(tf.float32, [None, self.config["sentence_length"], 5],
        #                                    name="word_addition")
        #self.char_addition = tf.placeholder(tf.float32, 
        #                                    [None, self.config["sentence_length"], self.config["word_length"], 4], 
        #                                    name="char_addition")
        self.sent_len = tf.placeholder(tf.int32, [None], name="sent_len")
        self.labels = tf.placeholder(tf.int32, [None, self.config["sentence_length"]], name ="labels")
        
    def _add_embdding(self):
        word_initializer = tf.constant(self.word_embed,
                                       dtype=tf.float32)
        char_initializer = tf.constant(self.char_embded,
                                       dtype=tf.float32)
                          
        with tf.variable_scope("embeded_layer"):
            word_embded = tf.get_variable("word_embeded", initializer=word_initializer)
            char_embded = tf.get_variable("char_embeded", initializer=char_initializer)
            #char_add_embded = tf.get_variable("char_add_embded",shape=[4]) 
            self.word_vectors = tf.nn.embedding_lookup(word_embded, self.sentence)            
            self.char_vectors = tf.nn.embedding_lookup(char_embded, self.word_list)
            #self.char_vectors = tf.concat(3, (self.char_vectors, self.char_addition))
            #self.word_vectors = tf.concat(2, (self.word_vectors, self.word_addition))

    #def _add_cnns_layer(self):
    #    filter_shape = [self.config["filter_size"], self.config["char_embded_size"] + 4, 
    #                    1, self.config["char_feature_size"]]
    #    self.W_cnns = tf.get_variable("W_cnns", filter_shape, tf.float32)
    #    self.b_cnns = tf.get_variable("b_cnns", [self.config["char_feature_size"]],tf.float32)
    #    tf.add_to_collection("loss", tf.nn.l2_loss(self.W_cnns) + tf.nn.l2_loss(self.b_cnns))
    
    #def _run_cnn(self, input):
    #    input = tf.expand_dims(input, -1)
    #    conv = tf.nn.conv2d(input, self.W_cnns, strides=[1,1,1,1], padding="VALID")
    #    h = tf.nn.relu(tf.nn.bias_add(conv, self.b_cnns))
    #    pooled = tf.nn.max_pool(h, 
    #                            ksize=[1, self.config["word_length"] - self.config["filter_size"] + 1, 1, 1],
    #                            strides=[1, 1, 1, 1], padding="VALID")
    #    return tf.squeeze(pooled,[1])

    @staticmethod
    def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
            b = tf.get_variable('b', [output_dim])

        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    
    def _tdnn(self, input_, scope="tdnn"):
        input_ = tf.expand_dims(input_, 1)
        reduced_length = self.config["word_length"] - self.config["filter_size"] + 1
        conv = self.conv2d(input_, 
                           self.config["char_feature_size"], 1, 
                           self.config["filter_size"], 
                           name="kernel")
        pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
        return pool

    def _add_model(self):
#        word_tensor = []
#        char_level_inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config["sentence_length"], self.char_vectors)]
#        self._add_cnns_layer()
#        for x in char_level_inputs:
#            word_tensor.append(self._run_cnn(x))
#        word_tensor = tf.concat(1, word_tensor)
        char_level_inputs = tf.reshape(self.char_vectors, 
                                       [-1, self.config["word_length"],self.config["char_embded_size"]])
        input_cnn = self._tdnn(char_level_inputs)
        word_tensor = tf.reshape(input_cnn, 
                                 [-1, self.config["sentence_length"], self.config["char_feature_size"]]
                                 )
        input = tf.concat(2, (self.word_vectors, word_tensor))
        
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config["rnn_hidden"], state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config["rnn_dropout"])
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                                                    tf.transpose(input, perm=[1,0,2]),
                                                    dtype=tf.float32, sequence_length=self.sent_len,
                                                    time_major=True)
        
        #output_fw = tf.reshape(tf.transpose(tf.pack(output_fw), perm=[1, 0, 2]), [-1, self.config["rnn_hidden"]])
        #output_bw = tf.reshape(tf.transpose(tf.pack(output_bw), perm=[1, 0, 2]), [-1, self.config["rnn_hidden"]])
     
        #W_fw = tf.get_variable('W_fw', [self.config["rnn_hidden"], self.config["num_class"]], tf.float32)
        #b_fw = tf.get_variable('b_fw', [self.config["num_class"]], tf.float32)
        #W_bw = tf.get_variable('W_bw', [self.config["rnn_hidden"], self.config["num_class"]], tf.float32)
        #b_bw = tf.get_variable('b_bw', [self.config["num_class"]], tf.float32)
        
        output = tf.concat(2, (output_fw, output_bw))
        output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]), [-1, 2 * self.config["rnn_hidden"]])
        W = tf.get_variable('W_fw', [2 * self.config["rnn_hidden"], self.config["num_class"]], tf.float32)
        b = tf.get_variable('b_fw', [self.config["num_class"]], tf.float32)
        predict = tf.nn.softmax(tf.matmul(output, W) + b)
        
        #predict_fw = tf.nn.softmax(tf.matmul(output_fw, W_fw) + b_fw)
        #predict_bw = tf.nn.softmax(tf.matmul(output_bw, W_bw) + b_bw)
        
        #tf.add_to_collection("loss", tf.nn.l2_loss(W_fw) + tf.nn.l2_loss(b_fw) +\
        #                     tf.nn.l2_loss(W_bw) + tf.nn.l2_loss(b_bw))
        self.prediction = tf.reshape(predict, 
                                     [-1, self.config["sentence_length"], self.config["num_class"]])
        #self.loss = self._cost()
        log_likehood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.prediction, self.labels, self.sent_len)
        self.loss = tf.reduce_mean(-log_likehood, name="loss")
        optimizer = tf.train.AdamOptimizer(0.003)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self.loss = tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.labels)
        #self.loss = tf.reduce_sum(self.loss, reduction_indices=1) / tf.cast(self.sent_len, tf.float32)
        #self.loss = tf.reduce_mean(self.loss)
        #self.label_predict = tf.argmax(self.prediction, axis=2)
        #optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        #self.train_op = optimizer.minimize(self.loss)

    def _cost(self):
        ce = self.labels * tf.log(self.prediction)
        ce = -tf.reduce_sum(ce, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.labels), reduction_indices=2))
        ce *= mask
        ce = tf.reduce_sum(ce, reduction_indices=1)
        ce /= tf.cast(self.sent_len, tf.float32)
        return tf.reduce_min(ce)

    def _build_graph(self):
        self._add_placeholder()
        self._add_embdding()
        self._add_model()

    def __init__(self, config, word_embded, char_embded):
        self.word_embed = word_embded
        self.char_embded = char_embded
        self.config = config
        self._build_graph()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()
    
    def save_model(self):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        save_path=saver.save(sess,"tmp/model-%s.ckpt" %now)

    def partial_fit(self, sentence, word_list, sent_len, labels ):
        feed_dict = {self.sentence : sentence,
                     self.word_list : word_list,
                     self.sent_len: sent_len,
                     self.labels: labels
        }
        cost, opt = self.sess.run((self.loss, self.train_op), feed_dict= feed_dict)
        return cost

    def calc_total_cost(self, sentence, word_list,sent_len, labels ):
        feed_dict = {self.sentence : sentence,
                     self.word_list : word_list,
                     self.sent_len: sent_len,
                     self.labels: labels
        } 
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def transform(self, sentence, word_list, sent_len):
        feed_dict = {self.sentence : sentence,
                     self.word_list : word_list,
                     self.sent_len: sent_len
        }
        logits, transition_params = self.sess.run([self.prediction, self.transition_params], 
                                                  feed_dict=feed_dict)
        viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
            logits, transition_params)
        return viterbi_sequence
