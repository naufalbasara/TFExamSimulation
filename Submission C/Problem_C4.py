# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open("./sarcasm.json", 'r') as f:
        sarcasm = json.load(f)

    for item in sarcasm:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    val_sentences = sentences[training_size:]
    val_labels = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)# YOUR CODE HERE
    tokenizer.fit_on_texts(train_sentences)

    training_seq = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(training_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    val_seq = tokenizer.texts_to_sequences(val_sentences)
    val_padded = pad_sequences(val_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    train_labels = np.array(train_labels).reshape(-1, 1)
    val_labels = np.array(val_labels).reshape(-1, 1)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu'),
        tf.keras.layers.AveragePooling1D(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy')>0.79 and logs.get('val_accuracy')>0.79:
                print("\nDesired accuracy reached, stop training")
                self.model.stop_training = True

    model.fit(train_padded, train_labels, epochs=50, validation_data=(val_padded, val_labels), callbacks=[CustomCallback()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
