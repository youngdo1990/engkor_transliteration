# -*- coding=utf-8 -*-

from string import punctuation
from seq2seq_att import *
import pandas as pd
import os
from tqdm import tqdm


def transliteration_dictionary():
    model = Transliterator()
    model.use_pretrained_model()
    df = pd.read_excel("../dictionary.xlsx")
    words = list(df["word"])[:100]
    hangulization = []
    punctuation = []
    for w in tqdm(words):
        transliteration = model.decode_sequence(w)
        hangulization.append(transliteration[0])
        punctuation.append(transliteration[1])
    df = pd.DataFrame(data=list(zip(words, hangulization, punctuation)), columns=["word", "hangulization", "punctuation"])
    df.to_excel("transliteration_dict.xlsx", index=False)
    return


if __name__ == "__main__":
    # model = Transliterator()
    # model.use_pretrained_model()
    # test_list = ['attention', 'tokenizer', 'transliterator', 'suddenly', 'mecab', 'adidas', 'nike', "python", "cpp", "react", "c++"]
    # for x in test_list:
    #    print(model.decode_sequence(x)) # input: attention
    transliteration_dictionary()