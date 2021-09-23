import sys
import re
import psycopg2
from pandas import Series
import datetime
import pandas as pd
from io import StringIO
from Heka_etl_deletewords import pre_process
# import nltk
# from nltk.corpus import stopwords


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def connect():
    try:
        conn = psycopg2.connect(**param_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"error:{error},param_dic:{param_dic}")
        sys.exit(1)
        return None
    return conn


def execute_sql(sql):
    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except Exception as error:
        print(f"error:{error}")
        cursor.close()
        conn.close()
        return None
    results = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    if (len(results)==0):
        results = 0
    return results


def max_id_number(column, table):
    sql = f"""SELECT COALESCE(max({column}),0)  FROM  {table} """
    max_id = execute_sql(sql)[0][0]
    # print(f"查询得到的最大ID是=={max_id}")
    return max_id


def df_to_dt(table, columns, df):
    output = StringIO()
    df.to_csv(output, index=False, header=False, mode="a", sep=";")
    f = StringIO(output.getvalue())
    conn = connect()
    cursor = conn.cursor()
    cursor.copy_from(f, table, sep=";", columns=columns)
    conn.commit()
    cursor.close()
    conn.close()


#数据库查询后，直接生成为dataframe格式
def database_to_pd(sql):
    conn = connect()
    results = pd.read_sql(sql, conn)
    conn.close()
    return results


"""
两种模式：
一种是简单模式simple，只去除;分号十分过分的符号，目的是为了导入数据库原始数据.
另一种是完全模式，去除各种复杂的怪异符号，目的是为了清洗数据.
默认为简单模式
"""
def clear_spec(mystr, type="simple"):
    if type == "simple":
        results1 = re.sub("""[;"]+""", "", mystr)
        # print(results1)
        #去除不可见字符
        results = re.sub("""[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+""", "", results1)
    if type == "full":
        results1 = re.sub("""[0-9!"#$%&()*+,-./:;<=>?@，。?★、￥…【】《》？“”‘'！[\\]^_`{|}~]+""", "", mystr)
        # print(results1)
        #去除不可见字符
        results = re.sub("""[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+""", "", results1)
    return results


#把列表转成空格隔开的句子
def list_sentence(mylist):
    sentence = ""
    for s in mylist:
        j = s.strip()
        if j != "’":
            sentence = sentence+" " + j
    result = sentence.strip()
    return result



#这里开始准备去除各种没有用的词，然后重新组成简洁的句子
def deletewords(x):
    # tokens = nltk.word_tokenize(x)
    # test_words = [word.lower() for word in tokens]
    # test_words_set = set(test_words)
    # filtered = [w for w in test_words_set if (w not in stopwords.words('english'))]
    # results = list_sentence(filtered)
    filtered = pre_process(x)
    results = list_sentence(filtered)
    return results


#比较两个dataframe数据集
def find_new_data(df1, df2):
    result = pd.merge(df1, df2, on=["suggestion"], how="left")
    # print("-" * 30)
    # print( df1["suggestion"][    df1["suggestion"].str.contains("stenosis and spondylosis")==True    ])
    # print("-"*30)
    result.drop(result[pd.notnull(result['id'])].index, inplace=True)
    result.drop(['id'], axis=1, inplace=True)
    # print(result.head())
    if result.empty:
        # print("没有新增数据，不需要导入")
        error = 1
    else:
        result.index = Series(range(len(result)))
        error = 0
    return result, error

def etl(my_filename, my_sort):
    path = 'd:/Python_scraping/'+my_filename+'.json'
    with open(path, mode='r', encoding='UTF-8') as file:
        myJson = file.read()
        file.close()
    df = pd.read_json(myJson)
    df.rename(columns={0: "suggestion_old"}, inplace=True)
    df["suggestion"] = df["suggestion_old"].apply(lambda x: clear_spec(x, type="simple"))


    #开始比较，将已经有了的数据从df中删除掉，只留下新的数据\
    df1 = df
    sql = """select id,suggestion from "SearchSuggestions"  """
    df2 = database_to_pd(sql)
    df3 = find_new_data(df1, df2)
    if df3[1] == 1:
        print("没有新增数据，不需要导入")
    else:
        df = df3[0]
        # print(df)
        df["suggestion_temp"] = df["suggestion_old"].apply(lambda x: clear_spec(x, type="full"))
        max_id_suggestion = max_id_number("id", '"SearchSuggestions"')
        id_add_suggestion = list(range(max_id_suggestion+1, max_id_suggestion+1+len(df.index)))
        df.insert(0, "id", id_add_suggestion)
        df["tokenize"] = df["suggestion_temp"].apply(lambda x: deletewords(x))
        df.drop(["suggestion_old", "suggestion_temp"], axis=1, inplace=True)
        df["sort"] = ""
        df['"createdAt"'] = datetime.datetime.today()
        df['"updatedAt"'] = datetime.datetime.today()
        df['sort'] = my_sort
        print(f"有{df.shape[0]}条数据导入")
        columns = df.columns
        df_to_dt('"SearchSuggestions"', columns, df)


def etl2(my_df, my_sort):
    my_df.rename(columns={0: "suggestion_old"}, inplace=True)
    my_df["suggestion"] = my_df["suggestion_old"].apply(lambda x: clear_spec(x, type="simple"))
    #开始比较，将已经有了的数据从df中删除掉，只留下新的数据\
    df1 = my_df
    sql = """select id,suggestion from "SearchSuggestions"  """
    df2 = database_to_pd(sql)
    df3 = find_new_data(df1, df2)
    if df3[1] == 1:
        print("没有新增数据，不需要导入")
    else:
        my_df = df3[0]
        # print(df)
        my_df["suggestion_temp"] = my_df["suggestion_old"].apply(lambda x: clear_spec(x, type="full"))
        max_id_suggestion = max_id_number("id", '"SearchSuggestions"')
        id_add_suggestion = list(range(max_id_suggestion+1, max_id_suggestion+1+len(df.index)))
        my_df.insert(0, "id", id_add_suggestion)
        my_df["tokenize"] = df["suggestion_temp"].apply(lambda x: deletewords(x))
        my_df.drop(["suggestion_old", "suggestion_temp"], axis=1, inplace=True)
        my_df["sort"] = ""
        my_df['"createdAt"'] = datetime.datetime.today()
        my_df['"updatedAt"'] = datetime.datetime.today()
        my_df['sort'] = my_sort
        print(f"有{my_df.shape[0]}条数据导入")
        columns = my_df.columns
        df_to_dt('"SearchSuggestions"', columns, my_df)


my_filename = "fbtext4"
my_sort = "4"
etl(my_filename, my_sort)