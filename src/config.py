image_width = 300
image_height = 300
image_channels = 3
BATCH_SIZE = 6
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
num_layers = 2
num_heads = 8
dff = 1024
top_k = 5000
vocab_size = top_k + 1
features_shape = 2048
attention_features_shape = 81
# data_path = '../../xraydata/data'
data_path = '../../xraydata/padchest'
EPOCHS = 20
print_every = 4000
max_length = 173
