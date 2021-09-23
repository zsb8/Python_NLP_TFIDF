import json
from sentence_transformers import SentenceTransformer
import Heka_utils as utils
import time
begin_time = int(time.time())

model = SentenceTransformer("stsb-distilbert-base")
physio_embeddings = dict()
sql = """select a.id, b.reasons from """\
    """(select b.id, a."firstName" as fn, a."lastName" as ln from "Clients" as a inner join  "Physios" as b """\
    """on a.id=b."ClientId" where a."firstName" is not null and a."lastName" is not null) as a """\
    """inner join (select a.id, string_agg(a.reasons,',') as reasons from """\
    """(select distinct "PhysioId" as id, expertise as reasons from "Expertises" """\
    """where "PhysioId" is not null and expertise is not null) as a """\
    """group by a.id order by a.id) as b on a.id=b.id where b.reasons<>'> Doesn’t apply' order by a.id"""
df = utils.database_to_pd(sql)
df['id'] = df['id'].apply(lambda x: int(x))
df['reasons'] = df['reasons'].str.split(',', expand=False)
df['result'] = df['reasons'].apply(lambda x: model.encode(x).tolist())
for row in df.itertuples():
    physio_embeddings[getattr(row, 'id')] = getattr(row, 'result')
print("completed")
with open("physio_embeddings.json", "w") as f:
    json.dump(physio_embeddings, f, indent=4)


end_time = int(time.time())
print(f"This task spend  {round((end_time - begin_time) / 60, 2)} minutes.")
