import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from flask import Flask, request
from flask_restful import Api, Resource
import datetime
import time, datetime
import json
from flask_cors import CORS, cross_origin

openai.api_key = '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'
HTTP_URL_PATTERN = r'^http[s]*://.+'
PORT = 5053
domain = "osmanlidogaltas.com"
full_url = "https://osmanlidogaltas.com/"


def remove_newlines(line):
    line = line.str.replace('\n', ' ')
    line = line.str.replace('  ', ' ')
    line = line.str.replace('  ', ' ')
    return line


texts = []
for file in os.listdir("text/" + domain + "/"):
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()
        texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

df = pd.DataFrame(texts, columns=['fname', 'text'])
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()
tokenizer = tiktoken.get_encoding("cl100k_base")
df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()
max_tokens = 500


def split_into_many(text, max_tokens=max_tokens):
    sentences = text.split('. ')
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):

        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


shortened = []
for row in df.iterrows():

    if row[1]['text'] is None:
        continue

    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    else:
        shortened.append(row[1]['text'])

df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()
df['embeddings'] = df.text.apply(
    lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()


def create_context(
        question, df, max_len=1800, size="ada"
):
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    returns = []
    cur_len = 0

    for i, row in df.sort_values('distances', ascending=True).iterrows():

        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
):
    start = time.time()
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"undefined\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        roundtrip = time.time() - start
        return [response["choices"][0]["text"].strip(), roundtrip]
    except Exception as e:
        print(e)
        return ""


app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)


class AIController(Resource):
    def get(self):
        now = datetime.datetime.now()
        if (request.args.get('query') is not None):
            answer = answer_question(df, question=request.args.get('query'))
            status = 1
            askedQuery = request.args.get('query')
        else:
            status = 0
            askedQuery = 'null'
            answer = ['null', 0]

        return {
            'data': {
                'status': status,
                'requestTime': answer[1],
                'askedQuery': askedQuery,
                'content': answer[0],
                'date': int(time.time() * 1000)
            }
        }


api.add_resource(AIController, '/api/getAnswer')

if __name__ == '__main__':
    app.run(port=PORT)
