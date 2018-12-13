import tensorflow as tf
from src.nn import nn_utils
from src.utils import Vocab, TextUtils, ImageUtils


class Model:
    def __init__(self):
        self.output_vocab_size = 1000
        self.word_dim = 128

        self.beam_width = 3
        self.attn_n_unit = 256
        self.decoder_lstm_n_unit = 256

        self.start_token_id = 10
        self.end_token_id = 10
        self.max_iter = 150

        # for placeholder
        self.word_embedding = None
        self.raw_sentences = None
        self.raw_img = None
        self.sentence_lens = None
        self.is_training = False
        self.target_weight = []

    def automatic_set(self, vocab, args):
        self.output_vocab_size = len(vocab)
        self.word_dim = args.get("word_dim",self.word_dim)

        self.beam_width = args.get("beam_width", self.beam_width)
        self.attn_n_unit = args.get("attn_n_unit", self.attn_n_unit)
        self.decoder_lstm_n_unit = args.get("decoder_lstm_n_unit", self.decoder_lstm_n_unit)

        self.start_token_id = vocab.get_id_from_word(Vocab.sos)
        self.end_token_id = vocab.get_id_from_word(Vocab.eos)
        self.max_iter = args.get("max_iter", self.max_iter)

        self.vocab = vocab

    def build_multi_graph(self):
        tf.reset_default_graph()

        self.train_graph = tf.Graph()
        self.infer_graph = tf.Graph()

        with self.train_graph.as_default():
            train_op, loss, correct, sample_ids, logits = self.build('training',True)
            initializer = tf.global_variables_initializer()

            train_saver = tf.train.Saver()
            self.train_stuff = {
                'train_op':train_op,
                'loss':loss,
                'correct':correct,
                'sample_ids':sample_ids,
                'logits':logits
            }

        with self.infer_graph.as_default():
            _, _, correct_test, pred_ids, _ = self.build('inference',False)

            infer_saver = tf.train.Saver()
            self.infer_stuff = {
                'correct':correct_test,
                'sample_ids':sample_ids
            }

        self.train_sess = tf.Session(graph=self.train_graph)
        self.infer_sess = tf.Session(graph=self.infer_graph)

        self.train_sess.run(initializer)

    def build(self, mode, not_build_placeholder=False):
        self._assert_mode(mode) #assert mode in ['training', 'inference']
        if not not_build_placeholder: self._build_placeholder()

        word_embedding = tf.get_variable("E", shape=[self.output_vocab_size, self.word_dim])
        embs = tf.nn.embedding_lookup(word_embedding, self.raw_sentences)

        memory, final_states = self._build_encoder(self.raw_img, self.is_training)
        logits, sample_ids = self._build_decoder(memory,final_states,word_embedding,self.raw_sentences,mode)

        correct, loss, train_op = None, tf.no_op(), None
        if mode == 'training':
            crossx = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.raw_target_sentences, logits=logits)
                #tf.losses.softmax_cross_entropy(tf.one_hot(self.raw_target_sentences, self.output_vocab_size), logits)
            loss = tf.reduce_mean(crossx * self.target_weight)
            train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

            correct = tf.reduce_sum(tf.cast(tf.equal(sample_ids, self.raw_target_sentences), dtype=tf.float32)) / self.max_iter

        return train_op , loss , correct , sample_ids , logits

    def _build_placeholder(self):
        batch_size = None # None mean differ between batches
        max_sen_len = None
        img_height, img_width = None, None

        self.raw_img = tf.placeholder(dtype=tf.int8, shape=[batch_size, img_height, img_width, 1]) #
        self.raw_sentences = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_sen_len], name='raw_sentences')
        self.raw_target_sentences = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_sen_len], name='raw_sentences')
        self.sentence_lens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='sentence_lens')
        self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        self.target_weight = tf.placeholder(dtype=tf.float32,shape=[batch_size, max_sen_len], name='target_weight')

    def _build_encoder(self, img, is_training, reuse=False):
        with tf.variable_scope("Encoder",reuse=reuse):
            output = tf.layers.conv2d(img - 0.5, 64, 3, 1,'SAME',activation=tf.nn.relu, name='conv1')
            output = tf.layers.max_pooling2d(output,2,2,name='pool1')

            output = tf.layers.conv2d(output, 128,3,1,'SAME',activation=tf.nn.relu,name='conv2')
            output = tf.layers.max_pooling2d(output,2,2,name='pool2')

            output = tf.layers.conv2d(output, 256, 3, 1, activation=None, name='conv3')
            output = tf.layers.batch_normalization(output, training=is_training)
            output = tf.nn.relu(output)

            output = tf.layers.conv2d(output, 256, 3, 1, activation=tf.nn.relu, name='conv4')
            output = tf.layers.max_pooling2d(output,[2,1],[2,1],name='pool3')

            output = tf.layers.conv2d(output, 512, 3,1, 'SAME', activation=None, name='conv5')
            output = tf.layers.batch_normalization(output,training=is_training)
            output = tf.nn.relu(output)
            output = tf.layers.max_pooling2d(output,[1,2],[1,2],name='pool4')

            output = tf.layers.conv2d(output, 512, 3, 1, 'SAME', activation=None, name='conv6')
            output = tf.layers.batch_normalization(output, training=is_training)
            output = tf.nn.relu(output) # (b,h,w,c)

            s = tf.shape(output)
            n_b,n_h,n_w,n_c = s[0], s[1], s[2], 512
            output = tf.reshape(output, shape=(-1, n_h * n_w, n_c))

            h = nn_utils.prepare_encoder_final_state(output, 'final_h', self.decoder_lstm_n_unit)
            c = nn_utils.prepare_encoder_final_state(output, 'final_c', self.decoder_lstm_n_unit)

        return output, tf.contrib.rnn.LSTMStateTuple(c,h)

    def _build_decoder(self, memory, init_states, word_embs, tgt_sent_lens, mode, reuse=False):
        self._assert_mode(mode) #assert mode in ['training', 'inference']

        with tf.variable_scope("Decoder", reuse=reuse) as decoder_scope:
            out_layer = tf.layers.Dense(self.output_vocab_size, use_bias=False)
            batch_size = tf.shape(memory)[0]

            cell, init_states = nn_utils.prepare_decoder_beam_stuff(memory, init_states, tgt_sent_lens, batch_size,
                                                                    self.beam_width, mode, self.attn_n_unit)

            if mode == 'training':
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=word_embs, sequence_length=tgt_sent_lens)
                decoder = tf.contrib.seq2seq.BasicDecoder( cell = cell, helper = helper, initial_state = init_states,
                                                           output_layer=out_layer)

                outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                                 maximum_iterations=self.max_iter,
                                                                                                 scope=decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id

            else:
                start_tokens = tf.tile(tf.constant([self.start_token_id], dtype=tf.int32), [batch_size])
                end_token = self.end_token_id

                beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell,
                                                               embedding = self.word_embedding,
                                                               start_tokens = start_tokens,
                                                               end_token = end_token,
                                                               initial_state = init_states,
                                                               beam_width = self.beam_width,
                                                               output_layer = out_layer )

                outputs, t1, t2 = tf.contrib.seq2seq.dynamic_decode(beam_decoder,
                                                                    maximum_iterations=self.max_iter, scope=decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id

        return logits, sample_ids

    def _create_feed_dict(self, images, normed_labels_in, normed_labels_out, labels_length, mode):
        self._assert_mode(mode) #assert mode in ['training','inference']
        feed_dicts = {}

        """
        self.raw_img = tf.placeholder(dtype=tf.int8, shape=[batch_size, img_height, img_width, 1]) #
        self.raw_sentences = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_sen_len], name='raw_sentences')
        self.raw_target_sentences = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_sen_len], name='raw_sentences')
        self.sentence_lens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='sentence_lens')
        self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        self.target_weight = tf.placeholder(dtype=tf.float32,shape=[batch_size, max_sen_len], name='target_weight')
        """

        feed_dicts[self.raw_img] = images
        feed_dicts[self.raw_sentences] = normed_labels_in
        feed_dicts[self.raw_target_sentences] = normed_labels_out
        feed_dicts[self.sentence_lens] = labels_length
        feed_dicts[self.is_training] = mode == 'training'
        feed_dicts[self.target_weight] = nn_utils.calculate_weight_from_label(normed_labels_out,self.vocab)

        return feed_dicts

    def _assert_mode(self, mode):
        assert mode in ['training','inference']

    def run(self, images, normed_labels_in, normed_labels_out, labels_length, mode):
        self._assert_mode(mode) #assert mode in ['training', 'inference']

        feed_dicts = self._create_feed_dict(images,normed_labels_in,normed_labels_out,labels_length,mode)
        loss, correct, sample_ids = None, None, None
        if mode == 'training':
            loss, _, correct, _ = self.train_sess.run([self.train_stuff['loss'],self.train_stuff['train_op'],
                                                    self.train_stuff['correct'], self.train_stuff['sample_ids']],
                                feed_dicts)

        else:
            sample_ids = self.infer_sess.run(self.infer_stuff['sample_ids'], feed_dicts)

        return loss, correct, sample_ids
