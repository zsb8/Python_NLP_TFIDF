import json
import csv
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

with open('fruit.json', mode='r', encoding='UTF-8') as file:
    myJson = file.read()
    file.close()
df = pd.read_json(myJson, orient="index")
text_list = list(df.index)

s_pos = nltk.pos_tag(text_list)
print(s_pos[0:10])

from nltk import ne_chunk
a = ne_chunk(s_pos)

from nltk.chunk import RegexpParser
grammer = '\n'.join([
    'NP: {<DT>*<NNP>}',
    'NP: {<JJ>*<NN>}',
    'NP: {<NNP>+}',
])
cp = RegexpParser(grammer)
result = cp.parse(s_pos)
print(result)
result.draw()
