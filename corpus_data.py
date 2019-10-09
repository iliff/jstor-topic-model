# imports
import json
from gensim import corpora, models

model_num = '02'  # change model number
# # load corpus data json
corpus_path = './gensim_output/' + model_num + '/'
with open(corpus_path + 'corpus_data_' + model_num + '.json') as json_file:
    data = json.load(json_file)

# load topic model
gensim_dictionary = corpora.Dictionary.load(corpus_path + 'gensim_dictionary_' + model_num + '.dict')
gensim_corpus = corpora.MmCorpus(corpus_path + 'gensim_corpus_' + model_num + '.mm')
lda_25 = models.ldamodel.LdaModel.load(corpus_path + 'lda_25_' + model_num + '.model')

# add top topics to doc in corpus data
i = 0
for doc in gensim_corpus[:10]:
    topic_tuples = []
    key = 'doc_' + str(i)
    topics = (lda_25.get_document_topics(doc, minimum_probability=0.2))
    # the following awkward code changes the np dtype from float32 to something that can be serialized by json
    for topic in topics:
        topic_num = topic[0]
        topic_prob = topic[1].item()  # change dtype from float32 to float
        topic_tuple = (topic_num, topic_prob)
        topic_tuples.append(topic_tuple)
    data[key]['topics'] = topic_tuples
    i += 1

print(data)

# with open(corpus_path + 'corpus_data_' + model_num + '.json', 'w') as json_file:
#     json.dump(data, json_file)
