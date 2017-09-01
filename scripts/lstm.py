# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

# Standard library
from os import path
import sys
import random
import json

# Third-party
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np

maxlen = 64
temp = 0.5

def get_model(maxlen, chars):
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True,
                   input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model

def train(corpus_file):
    corpus_file = path.abspath(corpus_file)
    cache_path = path.dirname(corpus_file)

    basename = path.basename(corpus_file).split("_corpus")[0]
    char_indices_file = path.join(cache_path,
                                  '{0}_char_indices.json'.format(basename))

    if not path.exists(char_indices_file):
        raise IOError("Character indices file doesn't exist: {0}"
                      .format(char_indices_file))

    # Path to corpus file
    with open(corpus_file) as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    with open(char_indices_file) as f:
        d = json.loads(f.read())
        chars = d['chars']
        char_indices = d['char_indices']

    # cut the text in semi-redundant sequences of maxlen characters
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = get_model(maxlen, chars)

    # train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)
        model.save_weights(join(cache_path,
                           'weights_{0:03d}.h5'.format(iteration)))

def sample(weights_file, seed, output_file, nchars):

    weights_file = abspath(weights_file)
    cache_path = dirname(weights_file)
    char_indices_file = join(cache_path, 'char_indices.json')
    corpus_file = join(cache_path, 'corpus.txt')

    with open(char_indices_file) as f:
        d = json.loads(f.read())
        chars = d['chars']
        char_indices = d['char_indices']
        indices_char = d['indices_char']

    with open(corpus_file) as f:
        text = f.read().lower()

    model = get_model(maxlen, chars)
    model.load_weights(weights_file)

    # def sample(a, temperature=1.0):
    #     # helper function to sample an index from a probability array
    #     a = np.log(a) / temperature
    #     a = np.exp(a) / np.sum(np.exp(a))
    #     return np.argmax(np.random.multinomial(1, a, 1))

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    generated = "seed:{}\n\n".format(seed)
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    sys.stdout.write(generated)
    sys.stdout.flush()

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(generated)

    i = 0
    while True:
        x = np.zeros((1, maxlen, len(char_indices)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = indices_char[str(next_index)]
        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if output_file is not None:
            with open(output_file, "a") as f:
                f.write(next_char)

        i += 1
        if nchars is not None and i >= nchars:
            break

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", dest="train", action="store_true")
    group.add_argument("--sample", dest="sample", action="store_true")

    parser.add_argument("--corpus", dest="corpus_file", type=str)
    parser.add_argument("--weights", dest="weights_file", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--output", dest="output_file", default=None)
    parser.add_argument("--nchars", dest="nchars", type=int, default=512)

    args = parser.parse_args()

    if args.train:
        train(args.corpus_file)

    elif args.sample:
        sample(weights_file=args.weights_file, seed=args.seed,
               output_file=args.output_file, nchars=args.nchars)
