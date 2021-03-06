# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

# Standard library
from os import path
import sys
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
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model

def train(corpus_file, cache_path, basename, char_indices_file):

    if not path.exists(char_indices_file):
        raise IOError("Character indices file doesn't exist: {0}"
                      .format(char_indices_file))

    # Path to corpus file
    with open(corpus_file) as f:
        text = f.read()#.lower()
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
    for iteration in range(1, 64+1):
        print()
        print('-' * 80)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)
        model.save_weights(path.join(
            cache_path, '{0}_weights_{1:03d}.h5'.format(basename, iteration)))

def sample(cache_path, basename, char_indices_file, weights_file,
           seed_text, output_file, nchars):

    weights_file = path.abspath(weights_file)

    with open(char_indices_file) as f:
        d = json.loads(f.read())
        chars = d['chars']
        char_indices = d['char_indices']
        indices_char = d['indices_char']

    model = get_model(maxlen, chars)
    model.load_weights(weights_file)

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # initialize the sampling with the seed text
    generated = seed_text
    sentence = generated

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
    parser.add_argument("--seed", dest="seed_text", type=str, default=None)
    parser.add_argument("--output", dest="output_file", default=None)
    parser.add_argument("--nchars", dest="nchars", type=int, default=512)

    args = parser.parse_args()

    if args.train:
        corpus_file = path.abspath(args.corpus_file)
        cache_path = path.dirname(corpus_file)
        basename = path.basename(corpus_file).split("_corpus")[0]

    else: # args.sample
        weights_file = path.abspath(args.weights_file)
        cache_path = path.dirname(weights_file)
        basename = path.basename(weights_file).split("_weights")[0]

    char_indices_file = path.join(cache_path,
                                  '{0}_char_indices.json'.format(basename))

    if args.train:
        train(corpus_file=corpus_file, cache_path=cache_path, basename=basename,
              char_indices_file=char_indices_file)

    elif args.sample:
        sample(cache_path=cache_path, basename=basename,
               char_indices_file=char_indices_file,
               weights_file=args.weights_file, seed_text=args.seed_text,
               output_file=args.output_file, nchars=args.nchars)
