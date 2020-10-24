# -*- coding: utf-8 -*-
"""
This module generates simple replys to emails in the same style as the user
based on their email history.

"""

!pip install PyDrive

!ls

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

download = drive.CreateFile({'id': '1kfUcDYHPdGkOyfhsYO2xPYQ9THStwdZu'})
download.GetContentFile('102623729836648532953-raw-reply-pairs.pkl')

download = drive.CreateFile({'id': '1V-ogRtNQ9xr5JglcPw-zAjp-O7mjzEva'})
download.GetContentFile('108371130441830548619-raw-reply-pairs.pkl')

import tensorflow as tf
import numpy as np
import pickle
import re
from collections import Counter
import string
from collections import defaultdict
import matplotlib.pyplot as plt
# %matplotlib inline

with open('102623729836648532953-raw-reply-pairs.pkl', 'rb') as f:
    data_102 = pickle.load(f)
with open('108371130441830548619-raw-reply-pairs.pkl', 'rb') as f:
    data_108 = pickle.load(f)

print('Number of records in 102:', len(data_102))
print('Number of records in 108:', len(data_108))
email_data=data_102

i=np.random.random_integers(len(email_data)-1)
print(i)
print(email_data[i][0]['message_core_body_sentences'])
print(email_data[i][1]['message_core_body_sentences'])

import requests
import json
import numpy as np


def _chunks(collection, chunk_size):
    chunk_size = max(1, chunk_size)
    return (collection[i:i + chunk_size] for i in range(0, len(collection), chunk_size))


class USEClient(object):
    def __init__(self, base_url="https://sentence-encoder.pointapi.com"):
        """ Create a new Universal Sentence Encoder Client
        :param base_url: the base URL of the server
        """
        self.endpoint = base_url + '/v1/models/USE:predict'

    def encode_text(self, text):
        """ Encode a piece of text.
        :param text: some text (str)
        :return: a np.array of shape (512,)
        """
        body = {"instances": [text]}
        r = requests.post(self.endpoint, data=json.dumps(body))
        return np.array(r.json()['predictions'][0])

    def encode_texts(self, texts, batch_size=1000):
        """ Encode a piece of text.
        Note: Requests will be sent in batches of maximum 1000 (`batch_size`)
        sentences per batch to avoid `413 Request Entity Too Large` error.
        :param texts: some texts ([str, ...])
        :param batch_size: how many sentences to put in a single request
        :return: a np.array of shape (?, 512)
        """
        results = []
        for batch in _chunks(texts, batch_size):
            body = {"instances": batch}
            r = requests.post(self.endpoint, data=json.dumps(body))
            results.append(np.array(r.json()['predictions']))

        return np.concatenate(results)

    def sentence_similarity(self, s0, s1):
        """ Compute sentence similarity
        :param s0: a sentence (str)
        :param s1: a sentence (str)
        :return: a float in range [0,1]
        """
        v0 = np.array(self.encode_text(s0))
        v1 = np.array(self.encode_text(s1))

        return np.dot(v0, v1) / (np.sqrt(v0.dot(v0)) * np.sqrt(v1.dot(v1)))
client = USEClient(base_url="https://sentence-encoder.pointapi.com")
sentences = ["Hello!", "How do you do?"]
encodings=client.encode_texts(sentences)

"""**Part 1: Finding common response phrases**"""

def count_letters(input_string, valid_letters=string.ascii_letters):
    count = Counter(input_string) 
    return sum(count[letter] for letter in valid_letters)
email_response=[[] for i in range(len(email_data))]
for i in range(len(email_data)):
  if (len(email_data[i][0]['message_core_body_sentences'])!=0):
    string_list=[re.sub('[#*<>?-]','',each_sent) for each_sent in email_data[i][0]['message_core_body_sentences']]
    pruned_str_list=[]    
    for each_sent in string_list:         
      each_sent=' '.join(each_sent.split())
      if ('thank' not in each_sent.lower()) and ('sincere' not in each_sent.lower()) and (count_letters(each_sent.lower())>3):
        pruned_str_list.append(each_sent)                        
    email_response[i]=pruned_str_list

i=np.random.random_integers(len(email_data)-1)
print(email_response[i])
print(email_data[i][0]['message_core_body_sentences'])

email_response_embeddings=[[] for i in range(len(email_response))]
for i in range(len(email_response)):
  if (len(email_response[i])!=0):
    email_response_embeddings[i]=client.encode_texts(email_response[i])

response_phrase_embedding=[]
response_phrase_indexing=[]
for i in range(len(email_response_embeddings)):
  if (len(email_response_embeddings[i])!=0):
    for j in range(len(email_response_embeddings[i])):
      response_phrase_embedding.append(email_response_embeddings[i][j])
      response_phrase_indexing.append([i,j])

print(len(response_phrase_embedding))
print(len(response_phrase_indexing))
response_phrase_indexing[23]

response_phrase_distance_matrix=np.zeros((len(response_phrase_embedding),len(response_phrase_embedding)))
for i in range(len(response_phrase_embedding)):
  for j in range(i+1,len(response_phrase_embedding)):
    response_phrase_distance_matrix[i][j]=np.linalg.norm(response_phrase_embedding[i]-response_phrase_embedding[j])

response_phrase_distance_vector=np.reshape(response_phrase_distance_matrix,len(response_phrase_embedding)*len(response_phrase_embedding))

distance_threshold=0.1
number_of_similar_phrases=np.sum(response_phrase_distance_vector<distance_threshold)-(response_phrase_distance_vector.shape[0]-len(response_phrase_embedding))/2+len(response_phrase_embedding)
print('number_of_similar_phrases for threshold ', distance_threshold,':', number_of_similar_phrases)
distance_threshold=0.2
number_of_similar_phrases=np.sum(response_phrase_distance_vector<distance_threshold)-(response_phrase_distance_vector.shape[0]-len(response_phrase_embedding))/2+len(response_phrase_embedding)
print('number_of_similar_phrases for threshold ', distance_threshold,':', number_of_similar_phrases)
distance_threshold=0.4
number_of_similar_phrases=np.sum(response_phrase_distance_vector<distance_threshold)-(response_phrase_distance_vector.shape[0]-len(response_phrase_embedding))/2+len(response_phrase_embedding)
print('number_of_similar_phrases for threshold ', distance_threshold,':', number_of_similar_phrases)

distance_threshold=0.4
response_phrase_distance_cluster=[]
response_phrase_distance_score=np.sum(np.logical_and((response_phrase_distance_matrix<distance_threshold),(response_phrase_distance_matrix>0)),axis=1,keepdims=True)
print(response_phrase_distance_score.shape)
plt.plot(response_phrase_distance_score) 
plt.show()

thres_freq=10
top_response_phrases_distance_index=np.where(response_phrase_distance_score>thres_freq)[0]
#print(top_response_phrases_index)
for i in range(len(top_response_phrases_distance_index)):
  k,l=response_phrase_indexing[top_response_phrases_distance_index[i]]
  print(email_response[k][l])

response_phrase_dotproduct_matrix=np.zeros((len(response_phrase_embedding),len(response_phrase_embedding)))
for i in range(len(response_phrase_embedding)):
  for j in range(i+1,len(response_phrase_embedding)):
    response_phrase_dotproduct_matrix[i][j]=np.dot(response_phrase_embedding[i],response_phrase_embedding[j])
    
response_phrase_dotproduct_vector=np.reshape(response_phrase_dotproduct_matrix,len(response_phrase_embedding)*len(response_phrase_embedding))

dot_threshold=0.7
number_of_similar_phrases=np.sum(response_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)
dot_threshold=0.8
number_of_similar_phrases=np.sum(response_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)
dot_threshold=0.9
number_of_similar_phrases=np.sum(response_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)

dot_threshold=0.9
response_phrase_cluster=[]
response_phrase_dotproduct_score=np.sum((response_phrase_dotproduct_matrix>dot_threshold),axis=1,keepdims=True)
print(response_phrase_dotproduct_score.shape)
plt.plot(response_phrase_dotproduct_score) 
plt.show()

unique_response_phrases_dotproduct=[]
thres_freq=10
top_response_phrases_dotprod_index=np.where(response_phrase_dotproduct_score>thres_freq)[0]
#print(top_response_phrases_index)
for i in range(len(top_response_phrases_dotprod_index)):
  k,l=response_phrase_indexing[top_response_phrases_dotprod_index[i]]
  unique_response_phrases_dotproduct.append(email_response[k][l])
  print(email_response[k][l])

print('Number of top response phrases through dot product metric before consolidation:',len(top_response_phrases_dotprod_index))
unique_response_phrases_dotproduct=list(set(unique_response_phrases_dotproduct))
print('Number of top response phrases through dot product metric after consolidation:',len(unique_response_phrases_dotproduct))
print('##########################################################################################################')

for unique_phrase in unique_response_phrases_dotproduct:
  print(unique_phrase)

print('Number of top response phrases through distance metric:',len(top_response_phrases_distance_index))

response_phrase_dotproduct_sorted_idx = np.argsort(-response_phrase_dotproduct_score.flatten())
plt.plot(np.sort(response_phrase_dotproduct_score.flatten())) 
plt.show()
print(response_phrase_dotproduct_sorted_idx)

"""**Part 2: Identifying trigger phrases**"""

email_trigger=[[] for i in range(len(email_data))]
for i in range(len(email_data)):
  if (len(email_data[i][1]['message_core_body_sentences'])!=0):
    string_list=[re.sub('[#*<>?-]','',each_sent) for each_sent in email_data[i][1]['message_core_body_sentences']]
    pruned_str_list=[]    
    for each_sent in string_list:         
      each_sent=' '.join(each_sent.split())
      if ('thank' not in each_sent.lower()) and ('sincere' not in each_sent.lower()) and (count_letters(each_sent.lower())>3):
        pruned_str_list.append(each_sent)                        
    email_trigger[i]=pruned_str_list

len(email_trigger)

i=np.random.random_integers(len(email_data)-1)
print(email_trigger[i])
print(email_data[i][1]['message_core_body_sentences'])

email_trigger_embeddings=[[] for i in range(len(email_trigger))]
for i in range(len(email_trigger)):
  if (len(email_trigger[i])!=0):
    email_trigger_embeddings[i]=client.encode_texts(email_trigger[i])

trigger_phrase_embedding=[]
trigger_phrase_indexing=[]
for i in range(len(email_trigger_embeddings)):
  if (len(email_trigger_embeddings[i])!=0):
    for j in range(len(email_trigger_embeddings[i])):
      trigger_phrase_embedding.append(email_trigger_embeddings[i][j])
      trigger_phrase_indexing.append([i,j])

print(len(trigger_phrase_embedding))
print(len(trigger_phrase_indexing))

trigger_phrase_dotproduct_matrix=np.zeros((len(trigger_phrase_embedding),len(trigger_phrase_embedding)))
for i in range(len(trigger_phrase_embedding)):
  for j in range(i+1,len(trigger_phrase_embedding)):
    trigger_phrase_dotproduct_matrix[i][j]=np.dot(trigger_phrase_embedding[i],trigger_phrase_embedding[j])
    
trigger_phrase_dotproduct_vector=np.reshape(trigger_phrase_dotproduct_matrix,len(trigger_phrase_embedding)*len(trigger_phrase_embedding))

dot_threshold=0.7
number_of_similar_phrases=np.sum(trigger_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)
dot_threshold=0.8
number_of_similar_phrases=np.sum(trigger_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)
dot_threshold=0.9
number_of_similar_phrases=np.sum(trigger_phrase_dotproduct_vector>dot_threshold)
print('number_of_similar_phrases for threshold ', dot_threshold,':', number_of_similar_phrases)

dot_threshold=0.9
trigger_phrase_cluster=[]
trigger_phrase_dotproduct_score=np.sum((trigger_phrase_dotproduct_matrix>dot_threshold),axis=1,keepdims=True)
print(trigger_phrase_dotproduct_score.shape)
plt.plot(trigger_phrase_dotproduct_score) 
plt.show()

unique_trigger_phrases_dotproduct=[]
thres_freq=10
top_trigger_phrases_dotprod_index=np.where(trigger_phrase_dotproduct_score>thres_freq)[0]
#print(top_trigger_phrases_index)
for i in range(len(top_trigger_phrases_dotprod_index)):
  k,l=trigger_phrase_indexing[top_trigger_phrases_dotprod_index[i]]
  unique_trigger_phrases_dotproduct.append(email_trigger[k][l])
  print(email_trigger[k][l])

print('Number of top trigger phrases through dot product metric before consolidation:',len(top_trigger_phrases_dotprod_index))
unique_trigger_phrases_dotproduct=list(set(unique_trigger_phrases_dotproduct))
print('Number of top trigger phrases through dot product metric after consolidation:',len(unique_trigger_phrases_dotproduct))
print('##########################################################################################################')

for unique_phrase in unique_trigger_phrases_dotproduct:
  print(unique_phrase)