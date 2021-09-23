# replace to use the new HEKA database
import json
from sentence_transformers import SentenceTransformer
import Heka_utils as utils
import time
begin_time = int(time.time())

model = SentenceTransformer("stsb-distilbert-base")
physio_embeddings = dict()
sql = f"select username.physio_id as id, reason.reasons from " \
    f"(select physio.id as physio_id, users.first_name as fn, users.last_name as ln from heka_user as users " \
    f"inner join  physio on users.id=physio.user_id " \
    f"where users.first_name is not null and users.last_name is not null) as username inner join " \
    f"(select a.physio_id, string_agg(a.reasons,',') as reasons from " \
    f"(select distinct physio_id, expertise as reasons from physio_expertise " \
    f"where physio_id is not null and expertise is not null) as a " \
    f"group by a.physio_id order by a.physio_id) as reason " \
    f"on username.physio_id=reason.physio_id " \
    f"where reason.reasons<>'> Doesn’t apply' order by username.physio_id "

df = utils.database_to_pd(sql)
df['id'] = df['id'].apply(lambda x: int(x))
df['reasons'] = df['reasons'].str.split(',', expand=False)
df['result'] = df['reasons'].apply(lambda x: model.encode(x).tolist())
for row in df.itertuples():
    physio_embeddings[getattr(row, 'id')] = getattr(row, 'result')
print("completed")
with open("physio_embeddings_new.json", "w") as f:
    json.dump(physio_embeddings, f, indent=4)


end_time = int(time.time())
print(f"This task spend  {round((end_time - begin_time) / 60, 2)} minutes.")
