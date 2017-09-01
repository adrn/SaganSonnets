""" Turn JSON file of poems into a single text file """

# Standard library
import os
from os import path
import json

def main(data_file, index=None):

    data_file = path.abspath(data_file)
    root_path = path.abspath(path.join(path.dirname(data_file), ".."))
    cache_path = path.join(root_path, 'cache')

    if not path.exists(cache_path):
        os.makedirs(cache_path)

    basename = path.splitext(path.basename(data_file))[0]
    corpus_file = path.join(cache_path, "{0}_corpus.txt".format(basename))
    char_indices_file = path.join(cache_path, "{0}_char_indices.json"
                                  .format(basename))

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    all_poems = []
    for key in data:
        all_poems.append(data[key])

    fulltext = "\n".join(all_poems)
    with open(corpus_file, "w") as f:
        f.write(fulltext)

    chars = sorted(list(set(fulltext.lower())))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    with open(char_indices_file, "w") as f:
        f.write(json.dumps({'chars': chars,
                            'char_indices': char_indices,
                            'indices_char': indices_char},
                           indent=4))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("--data", dest="data_file", type=str, required=True)
    args = parser.parse_args()

    main(args.data_file)
