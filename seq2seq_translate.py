from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


"""
bible: https://gigazine.net/news/20181012-language-translator-deep-learning/
"""

"""
Q: What is timestep in LSTM deep learning?
A:
LSTM は前タイムステップの隠れ状態を現タイムステップの隠れ状態に使うので層内の並列化が困難であった.
update: ConvS2S (Convolutional Sequence to Sequence) は単語列の処理を LSTM から CNN に置き換えることでタイムステップ方向の並列化を可能にしたモデル．
url: http://deeplearning.hatenablog.com/entry/convs2s
"""

"""
Sequence-to-sequence model with an attention mechanism

アーキテクチャ:
Encoder_Embeddingレイヤー: Word2vec
Encoder_Inputレイヤー
Encoder_隠れレイヤー: LSTM
Attentionレイヤー: Softmax
Decoder_Embeddingレイヤー: tanh
Decoder_Inputレイヤー
Decoder_隠れレイヤー: LSTM
Decoder_Outputレイヤー: Softmax
Decoder_Generatingレイヤー
"""

# _PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
# _UNK = "_UNK"
# _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

import os
path = os.getcwd()
data_path = path + "/seq2seq_translate_slackbot/" + "jpn.txt"
num_samples = 10000

"""
Step 1. Preprocessing
"""

def create_dictionary(data_path, num_samples):

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_texts = [] # english
    target_texts = [] # jaoanese
    # idx2word
    input_english_vocab = set() # english
    target_japanese_vocab = set() # jaoanese
    #
    for line in lines[: num_samples - 1]:
        # [0] English, [1] Japanese
        input_text, target_text = line.split("\t")
        target_text = _BOS + target_text + _EOS
        input_texts.append(input_text)
        target_texts.append(target_text)
        # english
        for word in input_text:
            if word not in input_english_vocab:
                input_english_vocab.add(word)
        # jaoanese
        for word in target_text:
            if word not in target_japanese_vocab:
                target_japanese_vocab.add(word)

    return input_texts, target_texts, input_english_vocab, target_japanese_vocab

input_texts, target_texts, input_english_vocab, target_japanese_vocab = create_dictionary(data_path, num_samples)


"""
Step 2. Embedding Layer: from Sentence to Number
# なぜ辞書のidxとwordをひっくり返すの？
"""

# {key=idx: val=English_word} sort by id
input_english_vocab = sorted(list(input_english_vocab))
target_japanese_vocab = sorted(list(target_japanese_vocab))
# the number of sample vocablary
encoder_vocab_size = len(input_english_vocab)
decoder_vocab_size = len(target_japanese_vocab)
# Define max length of encoder / decoder
max_encoder_seq_length = max([len(text) for text in input_texts])
max_decoder_seq_length = max([len(text) for text in target_texts])

# word2idx
# enumerate()で、インデックス番号, 要素の順に取得する
inverse_input_vocab = dict(
    [(word, id) for id, word in enumerate(input_english_vocab)])
inverse_target_vocab = dict(
    [(word, id) for id, word in enumerate(target_japanese_vocab)])


"""
Step 3. The dimension of Input / Output data for Keras LSTM
・入力のshape: shapeが(batch_size, timesteps, input_dim)の3階テンソル．
・出力のshape: return_sequencesの場合：shapeが(batch_size, timesteps, input_dim)の3階テンソル．
"""
# 文章数、文章の最大文字数、辞書の合計単語数の3次元配列
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, encoder_vocab_size),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, decoder_vocab_size),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, decoder_vocab_size),
    dtype='float32')

"""
1. encoder
2. decoder
3. context_vector
4. x-axis: time-step lstm
5. y-axis: multi-layer lstm

3 Gate and 2 Cell of lstm
Gate: input, output, forget
state_c = memory from previous step
state_h = output of memory cell
"""


"""
Step ４. Modeling
"""

"""
Time Step for LSTM Layer
"""

for pair_text_idx, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for timestep, word in enumerate(input_text):
        encoder_input_data[pair_text_idx, timestep, inverse_input_vocab[word]] = 1.
    # decoder_target_data is ahead of decoder_input_data by one timestep
    for timestep, word in enumerate(target_text):
        decoder_input_data[pair_text_idx, timestep, inverse_target_vocab[word]] = 1.
        if timestep > 0:
            # decoder_target_data will be ahead by one timestep（LSTM は前タイムステップの隠れ状態を現タイムステップの隠れ状態に使う）
            # decoder_target_data will not include the start character.
            decoder_target_data[pair_text_idx, timestep - 1, inverse_target_vocab[word]] = 1.


# def Seq2Seq_Model(encoder_vocab_size, decoder_vocab_size, NUM_HIDDEN_UNITS):

NUM_HIDDEN_UNITS = 256 # NUM_HIDDEN_LAYERS
BATCH_SIZE = 64
NUM_EPOCHS = 100

"""
Encoder Architecture
"""

encoder_inputs = Input(shape=(None, encoder_vocab_size))
encoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_state=True)
# x-axis: time-step lstm
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

"""
Decoder Architecture
"""
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_inputs = Input(shape=(None, decoder_vocab_size))
decoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_sequences=True, return_state=True)
# x-axis: time-step lstm
decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states) # Set up the decoder, using `encoder_states` as initial state.
decoder_softmax_layer = Dense(decoder_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

"""
Encoder-Decoder Architecture
"""
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy") # Set up model

    # return model


"""
Training our Seq2Seq Model
"""

# model = Seq2Seq_Model(encoder_vocab_size, decoder_vocab_size, NUM_HIDDEN_UNITS)
model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2) # Run training
model.save("seq2seq_translate_model.txt") # Save model

"""
After one hour or so on a MacBook CPU, we are ready for inference. To decode a test sentence, we will repeatedly:

1) Encode the input sentence and retrieve the initial decoder state
2) Run one step of the decoder with this initial state and a "start of sequence" token as target. The output will be the next target character.
3) Append the target character predicted and repeat.
"""

"""
# Next: inference mode .
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

"""

"""
Inference Setup (sampling)
Sample may refer to individual training examples.
A “batch_size” variable is hence the count of samples you sent to the neural network.
That is, how many different examples you feed at once to the neural network.
"""

# inputs=encoder_inputs, outputs=encoder_states
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
# State from encoder
decoder_state_input_h = Input(shape=(NUM_HIDDEN_UNITS,))
decoder_state_input_c = Input(shape=(NUM_HIDDEN_UNITS,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
# x-axis: time-step lstm
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
# inputsは日本語単語数の次元を持つレイヤーとencoderの記憶セル、アウトプットは初期化された記憶セルと
decoder_model = Model(inputs=[decoder_inputs] + decoder_state_inputs, outputs=[decoder_outputs] + decoder_states)


"""
謎
"""

# シーケンスを逆にして渡した方が精度が出る
input_idx2word = dict(
    (id, word) for word, id in inverse_input_vocab.items())
target_idx2word = deict(
    (id, word) for word, id in inverse_target_vocab.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, decoder_vocab_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(100):
    # Take one sequence (part of the training set) for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)

    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


"""
Sequence-to-sequence model with an attention mechanism

def encoder_decoder_model_with_attension(
        encoder_vocab_size,
        decoder_vocab_size,
        max_encoder_seq_length,
        max_decoder_seq_length):

    #  define length of encoder/decoder input
    encoder_input = Input(shape=(max_encoder_seq_length, ))
    decoder_input = Input(shape=(max_decoder_seq_length, ))

    # Encoder
    # we can add BatchNormalization, Masking
    encoder_input2embedding = Embedding(input_dim=encoder_vocab_size, output_dim=64, input_length=max_encoder_seq_length, mask_zero=True)(encoder_input)
    embedding2encoder = LSTM(units=64, return_sequences=True, dropout=0.3)(encoder_input2embedding)

    # Decoder
    # we can add BatchNormalization, Masking
    decoder_input2embedding = Embedding(input_dim=decoder_vocab_size, output_dim=64, input_length=max_decoder_seq_length, mask_zero=True)(decoder_input)
    embedding2decoder = LSTM(units=64, return_sequences=True, dropout=0.3)(decoder_input2embedding)

    # Attention
    encoder2attention = dot([embedding2decoder, embedding2encoder], axes=[2, 2])
    attention2softmax = Activation("softmax", name="attention")(encoder2attention)

    # Context
    thought_vector = dot([attention2softmax, embedding2encoder], axes=[2, 1])
    thought_vector2decoder = concatenate([thought_vector, embedding2decoder])

    # Attention Decoder
    decoder2tanh = Dense(units=64, activation="tanh")
    tanh2context = TimeDistributed(decoder2tanh)(thought_vector2decoder)
    context2softmax = Dense(units=decoder_vocab_size, activation="softmax")
    softmax2output = TimeDistributed(context2softmax)(tanh2context)

    # Attention Encoder-Decoder Model
    model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])

    return model

model = encoder_decoder_model_with_attension(encoder_vocab_size, decoder_vocab_size, max_encoder_seq_length, max_decoder_seq_length)
"""
