python preprocessing.py < data/train > ../work/train.clean
python preprocessing.py < data/dev > ../work/dev.clean
python preprocessing.py < data/test > ../work/test.clean
python generate_vocab.py
python generate_char_vocab.py
python data_config.py
