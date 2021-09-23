import sys
import psycopg2
import pandas as pd

def connect():
    try:
        conn = psycopg2.connect(**param_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"error:{error},param_dic:{param_dic}")
        sys.exit(1)
        return None
    return conn


def database_to_pd(my_sql):
    conn = connect()
    results = pd.read_sql(my_sql, conn)
    conn.close()
    return results


def list_sentence(my_list) -> str:
    """
    convert one list to a sentence
    :param my_list: list
    :return: str
    """
    sentence = ""
    for s in my_list:
        j = s.strip()
        if j != "’":
            sentence = sentence+" " + j
    result = sentence.strip()
    return result


def split_words(para: str, step: int) -> list:
    """
    input full sentence, split to some small sentences
    :param para:str    the full sentence of search result
    :param step:int    how many number words do you want to split in one sentence
    :return:list
    """
    sen = para.split(". ")
    word_all_list = []
    for i in range(len(sen)):
        wordlist = sen[i].split()
        word_list = [wordlist[j:j+step] for j in range(0, (len(wordlist)-step+1))]
        for n in word_list:
            one_sen = list_sentence(n)
            word_all_list.append(one_sen)
    return word_all_list


def execute_sql(sql):
    """
    Query any sql in one table
    :param sql: str
    :return: results:  list,[('A',), ('AA',), ('AAAIF',), ('AAALF',)]
    """
    conn_db = connect()
    cursor = conn_db.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn_db.commit()
    cursor.close()
    conn_db.close()
    return results


def list_drop_duplicates(old_list):
    """
    convert a list to a new list, drop all duplicates, and remain the sequence
    :param old_list: a list
    :return: a new list
    """
    data = old_list
    index = list(range(len(data)))
    columns = ["content"]
    df = pd.DataFrame(data, index, columns)
    df.drop_duplicates(subset=['content'], keep='first', inplace=True)
    new_list = list(df["content"])
    return new_list
