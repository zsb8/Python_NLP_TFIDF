import numpy as np
from collections import Counter
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import psycopg2
import sys
from Heka_etl_deletewords import pre_process

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def connect():
    try:
        conn = psycopg2.connect(**param_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"error:{error},param_dic:{param_dic}")
        sys.exit(1)
        return None
    return conn


def database_to_pd(sql):
    conn = connect()
    results = pd.read_sql(sql, conn)
    conn.close()
    return results

sql = """select id,suggestion,tokenize from "SearchSuggestions" """
df = database_to_pd(sql)
df["new_id"] = list(df.index)
docs = list(df["tokenize"])

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)


docs_words = [i.replace(",", "").split(" ") for i in docs]
chain = itertools.chain(*docs_words)
vocab = set(chain)
v2i = {v: i    for i, v in enumerate(vocab)}
i2v = {i: v    for i, v in enumerate(vocab)}

def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x

tf_methods = {
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x+1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }


def get_tf(method="boolean"):
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
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    b = weighted_tf(_tf)
    return b


def get_idf(method="prob"):
    words_all_number = len(i2v)
    df = np.zeros((words_all_number, 1))
    for i in range(words_all_number):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    b = idf_fn(df)
    return b


def get_keywords(n=2):
    for c in range(3):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        top_two_words = [i2v[i] for i in idx]
        print("doc{}, top{} {}".format(c, n, top_two_words))


def cosine_similarity(q, _tf_idf):
    _a = np.square(q)
    _b = np.sum(_a, axis=0, keepdims=True)
    _c = np.sqrt(_b)
    unit_q = q / _c
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(yourquestion, len_norm=False):
    q_words = yourquestion.replace(",", "").split(" ")
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)     # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]
    q_vec = q_tf * _idf
    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def list_sentence(mylist):
    sentence = ""
    for s in mylist:
        j = s.strip()
        if j != "’":
            sentence = sentence+" " + j
    result = sentence.strip()
    return result


tf = get_tf()           # [n_vocab, n_doc]
idf = get_idf()         # [n_vocab, 1]
tf_idf = tf * idf       # [n_vocab, n_doc]

yourquestion = "my back is very pain for some years."
question = list_sentence(pre_process(yourquestion))


scores = docs_score(question)
d_ids = scores.argsort()[-3:][::-1]
print(d_ids)
id_in_db = df["id"][(df["new_id"].apply(lambda x:True if x in d_ids else False)) == True]
docs_origin = list(df["suggestion"])
