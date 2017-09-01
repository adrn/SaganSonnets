# SaganSonnets

## Getting started

First, you have to generate a corpus from a JSON file containing the "units" of
text. This could be individual sonnets if you want to generate sonnets,
individual paragraphs, etc. In the below, you can put the `file.json` file
anywhere, but I recommend just putting it in a "data" directory in this project.
Other output (e.g., from training the model) will be written to a directory at
`data/../cache`, i.e. from the `<project root>/cache`:

    python scripts/make_corpus.py --data=data/limericks_all.json

Next you have to train the LSTM model:

    python scripts/lstm.py --train --corpus=cache/limericks_all_corpus.txt

Once you have a few "weights" files, you can start generating text:

    python scripts/lstm.py --sample \
    --weights=cache/file_weights_00X.h5 \
    --seed="There was" --nchars=100

## Dependencies

Some of the dependencies are listed in the `environment.yml` file. You can
create a new Anaconda environment set up to run these scripts with:

    conda create --file=environment.yml

---

## Authors

* Daniela Huppenkothen
* Adrian Price-Whelan
* Ellianna Schwab
* Erik Tollerud
