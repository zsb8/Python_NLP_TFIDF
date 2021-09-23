import os
import numpy as np
import pandas as pd
from collections import Counter
import itertools
from Heka_utils import database_to_pd


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x


def get_tf(method, vocab, docs, docs_words, v2i):
    number_word = len(vocab)
    number_doc = len(docs)
    _tf = np.zeros((number_word, number_doc), dtype=np.float64)
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            max_word = counter.most_common(1)[0][1]
            per_word = counter[v]
            a = per_word / max_word
            word_series = v2i[v]
            _tf[word_series, i] = a
    tf_methods = {
        "log": lambda x: np.log(1 + x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    b = weighted_tf(_tf)
    return b


def get_idf(method, docs, docs_words, i2v):
    words_all_number = len(i2v)
    df_temp = np.zeros((words_all_number, 1))
    for i in range(words_all_number):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df_temp[i, 0] = d_count
    idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x + 1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x + 1))),
        "len_norm": lambda x: x / (np.sum(np.square(x)) + 1),
    }
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    b = idf_fn(df_temp)
    return b


def db_tfidf(df):
    df["new_id"] = list(df.index)
    docs = list(df["tokenize"])
    docs_words = [i.replace(",", "").split(" ") for i in docs]
    chain = itertools.chain(*docs_words)
    vocab = set(chain)
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for i, v in enumerate(vocab)}
    tf = get_tf("log", vocab, docs, docs_words, v2i)
    idf = get_idf("log", docs, docs_words, i2v)
    tf_idf = tf * idf
    return df, tf_idf, idf, v2i, i2v


if __name__ == '__main__':
    path = "db.h5"
    if os.path.exists(path):
        my_store = pd.HDFStore(path)
        dset = my_store.get('db')
        df = dset.iat[0, 0]
        tf_idf = dset.iat[1, 0]
        idf = dset.iat[2, 0]
        v2i = dset.iat[3, 0]
        i2v = dset.iat[4, 0]
        my_store.close
    else:
        sql = """select id,suggestion,tokenize from "SearchSuggestions"  """
        df = database_to_pd(sql)
        temp = db_tfidf(df)
        db = pd.DataFrame(np.array(temp), columns=["db"])
        my_store = pd.HDFStore('db.h5')
        my_store.put(key='db', value=db)
        my_store.close
