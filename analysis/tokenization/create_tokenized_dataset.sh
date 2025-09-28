#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
source ~/.venvs/nichejepa/bin/activate

python analysis/tokenization/create_tokenized_dataset.py \
    --config_file ${CONFIG_FILE}