"""
tf.app.flags.FLAGSを使い、TensorFlowのPythonファイルを実行する際にパラメタを付与する
下記のようにすると、パラメタ付与が可能になり、デフォルト値やヘルプ画面の説明文を登録できる

tf.app.flags.DEFINE_string('変数名', 'デフォルト値', """説明文""")

tf.app.flags.DEFINE_stringの他に、tf.app.flags.DEFINE_boolean, tf.app.flags.DEFINE_integerがある


===================================================

コードサンプル (test.py)

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 100, """データ数""")
tf.app.flags.DEFINE_string('img_path', './img', """画像ファイルパス""")

def main(argv):
    print(FLAGS.data_num, FLAGS.img_path)

if __name__ == '__main__':
    tf.app.run()
"""

"""
Tensorflow: https://qiita.com/yanosen_jp/items/70e6d6afc36e1c0a3ef3
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 12500, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 12500, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./datas", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./datas", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    source_file = open(source_path,"r")
    target_file = open(target_path,"r")

    source, target = source_file.readline(), target_file.readline()
    counter = 0
    while source and target and (not max_size or counter < max_size):
      counter += 1
      if counter % 50 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()

      source_ids = [int(x) for x in source.split()]
      target_ids = [int(x) for x in target.split()]
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          data_set[bucket_id].append([source_ids, target_ids])
          break
      source, target = source_file.readline(), target_file.readline()
    return data_set

  def create_model(session, forward_only):
      model = seq2seq_model.Seq2SeqModel(
                  FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,
                  FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
                  FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                  forward_only=forward_only)

      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

      #if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        #add
      if not os.path.isabs(ckpt.model_checkpoint_path):
        ckpt.model_checkpoint_path= os.path.abspath(os.path.join(os.getcwd(), ckpt.model_checkpoint_path))
        #so far
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
      return model

#学習済みモデル、語彙ファイルを読み込み
sess = tf.Session()
model = create_model(sess, True)
model.batch_size = 1

in_vocab_path = os.path.join(FLAGS.data_dir,
                             "vocab_in.txt")
out_vocab_path = os.path.join(FLAGS.data_dir,
                             "vocab_out.txt" )

in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
_, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

#発話から応答を生成する
def decode(sent):
    '''seq2seqモデルによる応答生成
    引数 sent(str)：発話
    返値 out(str)：応答
    '''
    sentence = sent
    sentence = wakati(sentence)
    token_ids = data_utils.sentence_to_token_ids(sentence, in_vocab)

    bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])

    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)

    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    out = "".join([rev_out_vocab[output] for output in outputs])

    return out


def wakati(input_str):
        '''分かち書き用関数
        引数 input_str : 入力テキスト
        返値 m.parse(wakatext) : 分かち済みテキスト'''
        m = MeCab.Tagger('-Owakati')
        wakatext = input_str
        #print(m.parse(wakatext))
        return m.parse(wakatext)


#~~slackbot~~
#生成された応答をBotに発言させる
from slackbot.bot import respond_to     # @botname: で反応するデコーダ
from slackbot.bot import listen_to      # チャネル内発言で反応するデコーダ
from slackbot.bot import default_reply  # 該当する応答がない場合に反応するデコーダ


@respond_to('')                         #入力した全ての発言が対象
def mention_func(message):
    in_message = message.body["text"]   #Slack上でのユーザ発話の生テキストを取得
    out = decode(in_message)            #モデルに入力し応答を生成する
    message.reply(out)                  # Botに発言させる
