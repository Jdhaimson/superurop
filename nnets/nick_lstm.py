from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import re
from keras.preprocessing.sequence import pad_sequences
import string_expander
from random import randint
from keras.callbacks import ModelCheckpoint
import pickle

sols_path = "regex_sols_subs.txt"
prompts_path = "regex_prompts_subs.txt"

args = sys.argv[1:]
print(args)
if args and args[0]:
    print("yes")
    sols_path = "regex_sols.txt"
    prompts_path = "regex_prompts.txt"

new_i_s = []

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")


prompts = open(prompts_path).read().lower()

PAD_WORD = "<PAD>"
START_WORD = "<START>"
END_WORD = "<END>"
prompts = re.sub(r'[,.]', '', prompts)
words = list(set(prompts.split()))

words.insert(0, END_WORD) # END
words.insert(0, START_WORD) # Start
words.insert(0, PAD_WORD) # Pad

print(words)
vocab_size = len(words)


word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

pickle.dump(word_indices, open("charge_word_indices.p", "wb" ))
pickle.dump(indices_word, open("charge_indices_word.p", "wb" ))


query_maxlen = 100

def char_from_one_hot(one_hot_vector):
    char_index = [i for i, el in enumerate(one_hot_vector) if el == 1][0]
    return indices_char[char_index]

def vectorize_prompt_lines(prompt_lines):
    Xq = []
    for line in prompt_lines:
        xq = [word_indices[w] for w in line.split()]
        Xq.append(xq)
    return pad_sequences(Xq, maxlen=query_maxlen)


def vectorize_sols(sol_data):
    sol = []
    for line in sol_data.splitlines():
        x = [char_indices[w] for c in line.split()]
        sol.append(x)
    return pad_sequences(sol, maxlen=maxlen)

def np_array_to_prompt_string(prompt_np_array):
    out_str = ""
    ar = prompt_np_array
    for row in ar:
        for i in range(len(row)):
            if indices_word[row[i]] not in set([START_WORD, END_WORD, PAD_WORD]):
                out_str += indices_word[row[i]] + " "
    return out_str

prompt_lines = prompts.splitlines()
prompt_lines = [START_WORD + " " + line + " " + END_WORD for line in prompt_lines]
vectorized_prompts = vectorize_prompt_lines(prompt_lines)
all_prompts = []

text = open(sols_path).read()
#text = re.sub(r'[\xc2\x99\x98\xe2\x80\x9d\x9c]',"'", text)
print('corpus length:', len(text))

chars = set(text)
print(chars)
print('total chars:', len(chars))
chars = list(chars)
PAD_CHAR = "%"
START_CHAR = "_"
END_CHAR = "#"

chars.insert(0, END_CHAR) # END Char
chars.insert(0, START_CHAR) # Start Char
chars.insert(0, PAD_CHAR) # Start Char

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

pickle.dump(char_indices, open("charge_char_indices.p", "wb" ))
pickle.dump(indices_char, open("charge_indices_char.p", "wb" ))

print("hello",char_indices)
# cut the text in semi-redundant sequences of maxlen characters
text_lines = text.splitlines()
text_lines = [line + END_CHAR for line in text_lines]
maxlen = 10
step = 1
sentences = []
next_chars = []
indy_mappings = []
for k in range(0, len(text_lines)):
    text_line = text_lines[k]
    for i in range(-1*maxlen, len(text_line)-maxlen, step):
        before_string = ""
        end_index = i + maxlen
        if i < 0:
            before_string_chars = text_line[:end_index]
            before_string = START_CHAR + before_string_chars
        else:
            before_string = text_line[i: end_index]
        sentences.append(before_string)
        next_chars.append(text_line[end_index])
        all_prompts.append(vectorized_prompts[k])
        indy_mappings.append(k)
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen))
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1


all_prompts = np.array(all_prompts)
print(all_prompts)
print("wut", all_prompts.shape)
print("wut", X.shape)
print("wut", X[0].shape)
print("wut", y.shape)
print(vocab_size)
print("check padding working:")
for i in range(50):
    print(char_from_one_hot(y[i]))
for i in range(50):
    print([indices_char[j] for j in X[i]])

print([indices_char[j] for j in X[2]])
print(char_from_one_hot(y[2]))
print(char_from_one_hot(y[3]))


def group_together():
    X_real = []
    for i in range(len(indy_mappings)-1):
        mapping = indy_mappings[i]
        next_mapping = indy_mappings[i+1]
        X_sub_real = []
        for j in range(mapping, next_mapping):
            X_sub_real.append(X[j])
    return np.array(X_real)


# build the model: 2 stacked LSTM
print('Build model...')
PROMPT_HIDDEN_SIZE = 128
GEN_HIDDEN_SIZE = 128

prompt_rnn = Sequential()
prompt_rnn.add(Embedding(vocab_size, 50))
prompt_rnn.add(LSTM(256, return_sequences=True))
prompt_rnn.add(Dropout(0.2))
prompt_rnn.add(LSTM(128, return_sequences=False))

gen_rnn = Sequential()
gen_rnn.add(Embedding(len(chars), 50))
gen_rnn.add(LSTM(256, return_sequences=False))

model = Sequential()
model.add(Merge([prompt_rnn, gen_rnn], mode='concat'))
model.add(Dense(len(chars), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature=1.0):
    if temperature is 0:
        return np.argmax(a)
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def split_data(X, Y, P, test_size=0.25, random_state=0):
    random.seed(random_state)
    random_indicies = range(len(X))
    # rand_i = [range(rand_indy[i], rand_indy[i+1]) for i in range(len(rand_indy)-1)]
    # rand_i = [val for sublist in rand_i for val in sublist]
    random.shuffle(random_indicies)
    num_test_samples = int(test_size*len(X))
    random_test_indicies = random_indicies[:num_test_samples]
    random_train_indicies = random_indicies[num_test_samples:]
    X_train, X_test, Y_train, Y_test, P_train, P_test = [], [], [], [], [], []
    print(len(random_train_indicies))
    for rand_i in random_train_indicies:
        X_train.append(X[rand_i])
        Y_train.append(Y[rand_i])
        P_train.append(P[rand_i])

    for rand_i in random_test_indicies:
        X_test.append(X[rand_i])
        Y_test.append(Y[rand_i])
        P_test.append(P[rand_i])

    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
    P_train, P_test = np.array(P_train), np.array(P_test)

    return X_train, X_test, Y_train, Y_test, P_train, P_test


P = all_prompts
X_train, X_test, Y_train, Y_test, P_train, P_test = split_data(X, y, P)
print(Y_train.shape)
print(y.shape)
print(X_train.shape)
print(X.shape)
print(P.shape)
print(P_train.shape)

# train the model, output generated text after each iteration
weights_path = "regex_net_weights_backup.hdf5"
print("Loading weights...")
model.load_weights(weights_path)

for iteration in range(1, 50):
    print('-' * 50)
    print('Iteration', iteration)
    #model.fit([all_prompts, X], y, batch_size=128, nb_epoch=5)
    checkpointer = ModelCheckpoint(filepath="regex_net_weights.hdf5", verbose=1, save_best_only=True)
    model.fit([P_train, X_train], Y_train, batch_size=128, nb_epoch=20, validation_data=([P_test, X_test], Y_test), show_accuracy=True, callbacks=[checkpointer])
    loss, acc = model.evaluate([all_prompts, X], y, batch_size=128, show_accuracy=True)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    for k in range(2):
        rand_i = randint(0,len(X)-1)
        rand_indy = indy_mappings[rand_i]
        for diversity in [0, 0.5, 1.0]:
            print('----- diversity:', diversity)

            generated = ''

            # print('----- Generating with seed: "' + sentence + '"' + "\n")
            x_gen = X[rand_i]
            print(x_gen)
            print(np.array(x_gen))
            print(np.array(x_gen).shape)
            prompt_np_array = np.array([all_prompts[rand_i]])
            print(prompt_np_array)
            prompt = np_array_to_prompt_string(prompt_np_array)
            print(x_gen)

            for iteration in range(100):
                preds = model.predict([prompt_np_array, np.array([x_gen])], verbose=2)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                x_gen = np.append(x_gen, next_index)
                x_gen = np.delete(x_gen, 0)
                if next_char is "#":
                    break

            print("prompt:")
            print(string_expander.prompt_expand(prompt, rand_indy))
            print("gen_pretty:")
            print(string_expander.regex_expand(generated, rand_indy))
