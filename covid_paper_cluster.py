import json
import os
import csv
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import spacy
from spacy_langdetect import LanguageDetector

# Create sentence transformer object using pretrained model
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences,  make questions into list

corpus = ['What is known about transmission, incubation, and environmental stability?',
          'What do we know about COVID-19 risk factors?',
          'What do we know about vaccines and therapeutics?',
          'What do we know about virus genetics, origin, and evolution?',
          'What has been published about medical care?',
          'What do we know about non-pharmaceutical interventions?',
          'What has been published about ethical and social science considerations?',
          'What do we know about diagnostics and surveillance?',
          'What has been published about information sharing and inter-sectoral collaboration?',
          'Create summary tables that address relevant factors related to COVID-19?',
          'Create summary tables that address therapeutics, interventions, and clinical studies?',
          'Create summary tables that address risk factors related to COVID-19?',
          'Create summary tables that address diagnostics for COVID-19?',
          'Create summary tables that address material studies related to COVID-19?',
          'Create summary tables that address population studies related to COVID-19?',
          'Create summary tables that address models and open questions related to COVID-19?',
          'Create summary tables that address patient descriptions related to COVID-19'
          ]

corpus_embeddings = embedder.encode(corpus)

num_clusters = 6
clustering_model = KMeans(n_clusters=num_clusters, init='k-means++',
                          max_iter=100, n_init=1, verbose=0, random_state=3425)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

ques_embeding = []
for cluster in clustered_sentences:
    l = []
    for i in cluster:
        qe = embedder.encode(i)
        l.append(qe)
    ques_embeding.append(l)

w = csv.writer(open("covid_paper_cluster.csv", "w"))
w.writerow(['PaperId', 'ClusterID'])

nlp = spacy.load('en_core_web_md')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


def preprocessing(text):
    text = BeautifulSoup(text, "lxml").get_text()
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Keeping only letters
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    # Removing additional whitespaces
    text = re.sub(r" +", ' ', text)
    return text


def language_validation(text):
    doc = nlp(text)
    if doc._.language['score'] >= 0.95:
        return True
    else:
        return False


for i in os.listdir("./DataSet"):
    with open("./DataSet/" + str(i)) as jf:
        data = json.load(jf)
        # print(data.keys())
        print(data['paper_id'])

        cluster_number = None

        if len(data['abstract']) != 0:
            whole_abstract = ""
            for j in data['abstract']:
                whole_abstract = whole_abstract + " " + j['text']
            print(whole_abstract)

            if language_validation(whole_abstract):
                preprocess_whole_abstract = preprocessing(whole_abstract)

                abstract_embeding = embedder.encode(preprocess_whole_abstract)

                cosine_result = []
                for cluster in ques_embeding:
                    l = []
                    for sub_ques in cluster:
                        cosine_scores = util.pytorch_cos_sim(sub_ques, abstract_embeding)
                        l.append(cosine_scores)
                    cosine_result.append(l)

                print(cosine_result)

                cluster_max_Value = []
                for i in cosine_result:
                    cluster_max_Value.append(max(i))

                cluster_number = cluster_max_Value.index(max(cluster_max_Value))
                print("This cluster number is ", cluster_number)

        else:
            print("No abstract")

        w.writerow([data['paper_id'], cluster_number])
