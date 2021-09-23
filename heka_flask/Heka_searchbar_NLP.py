from Heka_utils import list_sentence, split_words, database_to_pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def match_sentences(question: str,
                    docs: list,
                    num: int) -> list:
    """
    use sklearn to find the most match sentences
    :param question: str    the sentence you typed in the search bar
    :param docs: list  sentences stored in the 'SearchSuggestion' table
    :param num: int   you want to get the result number
    :return: list    the result sentence
    """
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(docs)
    qtf_idf = vectorizer.transform([question])
    res = cosine_similarity(tf_idf, qtf_idf)
    res = res.ravel().argsort()[~num+1:]
    result = []
    for i in res:
        result.append(docs[i])
    return result


def result_sentence(
        question: str,
        mode: str,
        step: int) -> list:
    """
    main program
    :param your_question: str    input the sentence in the search bar
    :param mode: str   "long" mode is show full sentences, "short" mode is show short segments
    :param step: int   how many words in one sentence showing in the frond-end
    :return:
    """
    sql = """select suggestion from "SearchSuggestions"  """
    df = database_to_pd(sql)
    docs = df["suggestion"]
    result_long = match_sentences(question, docs, 3)
    if mode == "long":
        return result_long
    if mode == "short":
        sen = question.split()
        if len(sen) > step:
            q = list_sentence(sen[~step + 1:])
        else:
            q = question
        result_short = []
        for i in result_long:
            if len(i.split()) > step:
                data = split_words(i, step)   # list
                result = match_sentences(q, data, 1)
                result_short.extend(result)
            else:
                result_short.append(i)
        return result_short
