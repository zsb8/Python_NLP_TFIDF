import psycopg2
import pandas as pd
import time
import boto3
from urllib.parse import unquote_plus
import json
import uuid
from sentence_transformers import SentenceTransformer


def database_to_pd(my_sql):
    conn = psycopg2.connect(**param_dic)
    results = pd.read_sql(my_sql, conn)
    conn.close()
    return results


def main():
    begin_time = int(time.time())
    model = SentenceTransformer("stsb-distilbert-base")
    physio_embeddings = dict()
    sql = """select a.id, b.reasons from """\
        """(select b.id, a."firstName" as fn, a."lastName" as ln from "Clients" as a inner join  "Physios" as b """\
        """on a.id=b."ClientId" where a."firstName" is not null and a."lastName" is not null) as a """\
        """inner join (select a.id, string_agg(a.reasons,',') as reasons from """\
        """(select distinct "PhysioId" as id, expertise as reasons from "Expertises" """\
        """where "PhysioId" is not null and expertise is not null) as a """\
        """group by a.id order by a.id) as b on a.id=b.id where b.reasons<>'> Doesn’t apply' order by a.id limit 3  """
    df = database_to_pd(sql)
    df['id'] = df['id'].apply(lambda x: int(x))
    df['reasons'] = df['reasons'].str.split(',', expand=False)
    df['result'] = df['reasons'].apply(lambda x: model.encode(x).tolist())
    for row in df.itertuples():
        physio_embeddings[getattr(row, 'id')] = getattr(row, 'result')
    print("completed")
    end_time = int(time.time())
    print(df.head())
    with open("/tmp/physio_embeddings.json", "w") as f:
        json.dump(physio_embeddings, f, indent=4)
    print(f"This task spend  {round((end_time - begin_time) / 60, 2)} minutes.")


s3_client = boto3.client('s3')


def lambda_handler(event, context):
    for record in event['Records']:
        key = unquote_plus(record['s3']['object']['key'])
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
        print(f"download_path={download_path}")
        upload_path = '/tmp/resized-{}'.format(tmpkey)
        print(f"upload_path={upload_path}")
        main()
        s3_client.upload_file('/tmp/physio_embeddings.json', 'heka.ca-flask-bucket2', key)
