import tensorflow as tf
from models.transformer import MultiHeadAttention, TransformerLayerWrapper, AoaMultiHeadAttention, AoaMultiHeadAttentionWrapper
from models.transformer import create_padding_mask

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class MultiheadDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(MultiheadDecoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.multiheadattention = MultiHeadAttention(self.units, num_heads= 8)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model

        features, _ = self.multiheadattention(features, features, features)
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights
        
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class TranslayerDecoder(tf.keras.Model):
    def __init__(self, num_layers, embedding_dim, units, num_heads ,dff , vocab_size):
        super(TranslayerDecoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.transformer_layers = TransformerLayerWrapper(num_layers= num_layers,
                                                        d_model= embedding_dim,
                                                        num_heads= num_heads,
                                                        dff = dff,
                                                        use_image= True
                                                        )
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        features, _ = self.transformer_layers(features, training = True)
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights
        
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class AoaDecoder(tf.keras.Model):
    def __init__(self,num_layers, embedding_dim, units, vocab_size):
        super(AoaDecoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.refine = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.encoder_aoa = AoaMultiHeadAttentionWrapper(num_layers,self.units, num_heads= 8)
        self.decoder_aoa = AoaMultiHeadAttentionWrapper(num_layers,self.units, num_heads= 8)

    def call(self, x, prev_state, features, hidden):
        # defining attention as a separate model
        features ,_ ,_ = self.encoder_aoa(features,features,features)
        # context_vector, attention_weights = self.attention(features, hidden)
        
        # mean pooling
        context_vector = tf.reduce_mean(features,axis = -1)
        context_vector = self.refine(context_vector)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_mask = create_padding_mask(x)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = [embed(token_i) , c_t-1 + meanpooling(a)]
        x = tf.concat([tf.expand_dims(context_vector + tf.reshape(prev_state,[prev_state.shape[0],-1]), 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # # shape == (batch_size, max_length, hidden_size)
        # x = self.fc1(output)
        out, prev_state, attention_weights = self.decoder_aoa(features, features, output, dec_mask) 
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(out, (-1, out.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, prev_state, state, attention_weights
        
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))