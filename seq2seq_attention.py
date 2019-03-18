
class Attention(tf.keras.Model):

    def __init__(self, units):
        # initialize initial values
        super(Attention, self).__init__()
        # architecture of nn
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hudden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        prob_score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(prob_score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
