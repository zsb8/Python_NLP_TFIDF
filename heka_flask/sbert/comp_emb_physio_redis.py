from sentence_transformers import SentenceTransformer
import Heka_utils as utils
import time
import redis
import pickle
begin_time = int(time.time())
rs = redis.StrictRedis(host='192.168.50.1111')
model = SentenceTransformer("stsb-distilbert-base")
physio_embeddings = dict()
# get data from postgres, cost 2 seconds
sql1 = """select a.id, b.reasons from """\
    """(select b.id, a."firstName" as fn, a."lastName" as ln from "Clients" as a inner join  "Physios" as b """\
    """on a.id=b."ClientId" where a."firstName" is not null and a."lastName" is not null) as a """\
    """inner join (select a.id, string_agg(a.reasons,',') as reasons from """\
    """(select distinct "PhysioId" as id, expertise as reasons from "Expertises" """\
    """where "PhysioId" is not null and expertise is not null) as a """\
    """group by a.id order by a.id) as b on a.id=b.id where b.reasons<>'> Doesn’t apply' order by a.id  """
df = utils.database_to_pd(sql1)
end_time = int(time.time())
print(f"This task1 spend  {round((end_time - begin_time), 2)} seconds.")

# calculate emb, cost 16 minutes
df['id'] = df['id'].apply(lambda x: int(x))
df['reasons'] = df['reasons'].str.split(',', expand=False)
df['result'] = df['reasons'].apply(lambda x: pickle.dumps(model.encode(x).tolist()))
df.drop('reasons', axis=1, inplace=True)
end_time = int(time.time())
print(f"This task2 spend  {round((end_time - begin_time), 2)} seconds.")

# store into redis database, cost 7 seconds
list_id = df['id']
list_result = df['result']
dict_result = dict(zip(list_id, list_result))
rs.hmset("emb_physio", dict_result)
end_time = int(time.time())
print(f"This task3 spend  {round((end_time - begin_time), 2)} seconds.")

# SQl phyiso's name data from postgres and input result to redis, cost 1 seconds
sql2 = """select b.id, a."firstName"||' '||"lastName" as name from "Clients" as a inner join  "Physios" as b """ \
    f"""on a.id=b."ClientId" where a."firstName" is not null and a."lastName" is not null  """\
    f"""  and a."firstName" not in ('First','frist2','first three','temp') and a."lastName" not in ('a')     """
result = utils.execute_sql(sql2)
physio_dict = dict(result)
rs.hmset("physio_name", physio_dict)
end_time = int(time.time())
print(f"This task4 spend  {round((end_time - begin_time), 2)} seconds.")

