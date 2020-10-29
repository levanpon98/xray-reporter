import time
import datetime
import math 

import tensorflow as tf
from tensorflow.keras.utils import Progbar

import config
from loader import load_data
from models.encoder import Encoder
from models.decoder import Decoder, MultiheadDecoder, TranslayerDecoder

if __name__ == '__main__':

    train_ds, valid_ds, max_length_train, max_length_valid, tokenizer = load_data(config.data_path)

    encoder = Encoder(config.embedding_dim)
    # decoder = Decoder(config.embedding_dim, config.units, config.vocab_size)
    # decoder = MultiheadDecoder(config.embedding_dim, config.units, config.vocab_size)
    decoder = TranslayerDecoder(num_layers= config.num_layers,
                                embedding_dim= config.embedding_dim, 
                                units= config.units,
                                num_heads= config.num_heads,
                                dff= config.dff,
                                vocab_size= config.vocab_size
                                )
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_path = "/content/drive/My Drive/xray-reporter/padchest/checkpoints"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    print("Restore latest checkpoints ...")
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    current_time = datetime.datetime.now().strftime('%Y%m%d-H%M%S')
    train_log_dir = '/content/drive/My Drive/xray-reporter/padchest/logs/gradient_tape/' + current_time + '/train'
    test_log_dir = '/content/drive/My Drive/xray-reporter/padchest/logs/gradient_tape/' + current_time + '/test'
    train_summary = tf.summary.create_file_writer(train_log_dir)
    test_summary = tf.summary.create_file_writer(test_log_dir)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
        # target length x units
        with tf.GradientTape() as tape:
            # 81 x 256
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                
                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)


        total_loss = (loss / int(target.shape[1]))
 
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        if math.isnan(loss):
          print(predictions)
          print(target)
          exit()
        return loss, total_loss

    def evaluate_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            predicted_id = tf.random.categorical(predictions, 1).numpy()
            dec_input = tf.expand_dims(predicted_id[:,0], 1)
  
        total_loss = (loss / int(target.shape[1]))

        return loss, total_loss

    EPOCHS = config.EPOCHS

    for epoch in range(0, EPOCHS):
        start = time.time()
        total_loss = 0

        # pb_i = Progbar(len(train_ds), stateful_metrics=['loss'])
        # Training
        print('[TRAIN] epoch',epoch + 1)
        for (batch, (img_tensor, target)) in enumerate(train_ds):

            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            # pb_i.add(config.BATCH_SIZE, values=[('total loss', total_loss)])
            # pb_i.add(config.BATCH_SIZE, values=[('batch loss', batch_loss)])
            if batch % config.print_every:
                print("avg loss = {} , total loss = {}".format(total_loss/batch, total_loss))
        print("End epoch",epoch + 1)
        print("avg loss = {} , total loss = {}".format(total_loss/batch, total_loss))
        ckpt_manager.save()

        # Evaluate
        print('[EVALUATE]')
        for (batch, (img_tensor, target)) in enumerate(valid_ds):
            batch_loss, t_loss = evaluate_step(img_tensor, target)
            total_loss += t_loss
            if batch % config.print_every:
                print("avg loss = {} , total loss = {}".format(total_loss/batch, total_loss))
        print("End epoch",epoch + 1)
        print("avg loss = {} , total loss = {}".format(total_loss/batch, total_loss))


