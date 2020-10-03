import tensorflow as tf 
from models.aoa import AoaLayer

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class AoaMultiHeadAttentionWrapper(tf.keras.layers.Layer):
  '''
  A wrapper of AoaMultiHeadAttention
  '''
  def __init__(self ,num_layers ,d_model ,num_heads , dropout_aoa =0.3):
    super(AoaMultiHeadAttentionWrapper, self).__init__()
    self.num_layers = num_layers

    self.aoa_layers = [AoaMultiHeadAttention(d_model, num_heads,dropout_aoa) 
              for _ in range(num_layers)]

  def call(self,v, k ,q, mask = None):
    '''
    For paper, num_layers = 6
    '''
    self_attn_weights = {}
    for i in range(self.num_layers):
      state, q, self_attn = self.aoa_layers[i](v,k,q, mask = mask)
      self_attn_weights['layer_{}'.format(i+1)] = self_attn
    return q ,state, self_attn_weights
 
class AoaMultiHeadAttention(tf.keras.layers.Layer):
  '''
  A wrapper of MultiheadAttention
  because Python use shallow copy so it may not true for residule connection
  '''
  def __init__(self, d_model, num_heads,dropout_aoa =0.3):
    super(AoaMultiHeadAttention, self).__init__()
    
    self.multiheadattention = MultiHeadAttention(d_model, num_heads= 8)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.aoa_layer = AoaLayer(d_model, dropout_aoa)
    
  def call(self,v, k ,q, mask = None):
    # q in this scope hasn't go through an linear layer
    out, attention_weights = self.multiheadattention(v, k, q, mask)
    state = self.aoa_layer(q,out)
    out = self.layernorm(q + state)
    return state,out, attention_weights

    
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
  
  def split_heads(self, x, batch_size):
    """
    Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask = None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

  def reset_state(self, batch_size):
      return tf.zeros((batch_size, self.d_model))

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

class FullyConnected(tf.keras.layers.Layer):
  def __init__(self, d_model, dff = 2048):
    super(FullyConnected, self).__init__()

    self.dense1 =  tf.keras.layers.Dense(dff, activation='relu') # (batch_size, seq_len, dff)
    self.dense2 =  tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  def call(self, x ):
    return self.dense2(self.dense1(x))

class TransformerLayerWrapper(tf.keras.layers.Layer):
  def __init__(self,num_layers ,d_model , num_heads, dff, rate=0.1, use_image = False):
    super(TransformerLayerWrapper, self).__init__()
    self.num_layers = num_layers
    self.trans_layers = [TransformerLayer(d_model, num_heads, dff, rate) 
                  for _ in range(num_layers)]
    self.use_image = use_image
  def call(self,x ,training):
    if not self.use_image:
      mask = create_padding_mask(x)
    else:
      mask = None
    self_attn_weights= {}
    for i in range(self.num_layers):
      x, self_attn = self.trans_layers[i](x, training = training,mask = mask)
      self_attn_weights['layer_{}'.format(i+1)] = self_attn
    return x,self_attn_weights

class TransformerLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1, with_external = False):
    super(TransformerLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = FullyConnected(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
    self.with_external = with_external  
    # if self.with_external:

  def call(self, x, training, mask):
    attn_output, self_attn = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)

    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2, self_attn

class Transformer(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    self.enc_layers = [TransformerLayer(d_model, num_heads, dff, rate) 
                      for _ in range(num_layers)]
    self.enc_layers = [TransformerLayer(d_model, num_heads, dff, rate, with_external=True) 
                      for _ in range(num_layers)]

