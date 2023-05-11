from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import keras.backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np

def cbow_model(vocab_size, embedding_dim, window_size):
    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=window_size*2))#embeddings_initializer='glorot_uniform'
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
    cbow.add(Dense(vocab_size, activation='softmax'))#kernel_initializer='glorot_uniform'
    cbow.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #cbow.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy', metrics=['accuracy'])
    return cbow

def unique_words(words_list):
    unique_words = set()
    for word in words_list:
        unique_words.add(word)
    return list(unique_words)

from keras.preprocessing import sequence

# Prepare the data for the CBOW model
def generate_data_cbow(corpus, window_size, V):
    all_in = []
    all_out = []

    # Iterate over all sentences
    for sentence in corpus:
        L = len(sentence)
        for index, word in enumerate(sentence):
            start = index - window_size
            end = index + window_size + 1

            # Empty list which will store the context words
            context_words = []
            for i in range(start, end):
                # Skip the 'same' word
                if i != index:
                    # Add a word as a context word if it is within the window size
                    if 0 <= i < L:
                        context_words.append(sentence[i])
                    else:
                        # Pad with zero if there are no words 
                        context_words.append(0)
            # Append the list with context words
            all_in.append(context_words)

            # Add one-hot encoding of the target word
            all_out.append(to_categorical(word, V))
                 
    return (np.array(all_in), np.array(all_out))
