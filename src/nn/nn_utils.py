import tensorflow as tf
from src.utils import Vocab
import numpy as np

def _make_tile(input, n):
    return tf.contrib.seq2seq.tile_batch(input, multiplier = n)

def prepare_decoder_beam_stuff(memory, fn_states, tgt_sen_lens, batch_size, beam_w, mode, attn_n_unit):
    assert mode in ['training', 'inference']

    tmp_memory = memory
    if mode is 'training': pass
    else:
        memory = _make_tile(memory, beam_w)
        tgt_sen_lens = _make_tile(tgt_sen_lens, beam_w)
        fn_states = _make_tile(fn_states, beam_w)

        batch_size = batch_size * beam_w

    attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=attn_n_unit, memory=memory, normalize=True,
                                                          memory_sequence_length=tgt_sen_lens)

    cell = tf.contrib.rnn.BasicLSTMCell(attn_n_unit)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mechanism, attention_layer_size=attn_n_unit,name="attention")
    decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=tmp_memory)

    return cell, decoder_initial_state

def prepare_encoder_final_state(memory, scope, n_unit, reuse=False):
    #memory: (batch_size, height * width, n_channel)
    with tf.variable_scope(scope,reuse=reuse):
        mean = tf.reduce_mean(memory, axis=1) # (batch_size, n_channel)
        mean = tf.layers.dense(mean, n_unit, activation=tf.nn.tanh)

    return mean

def calculate_weight_from_label(labels, vocab):
    weights = np.ones_like(labels)
    weights[weights == vocab.get_id_from_word(Vocab.pad)] = 0.0

    return weights