import os
import json
from gensim import corpora, models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_gensim_dictionary(docs, no_below, no_above, model_path, model_name):
    gensim_dictionary = corpora.Dictionary(docs)
    gensim_dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    gensim_dictionary.save(model_path + '/' + model_name + '.dict')
    return gensim_dictionary


def make_gensim_corpus(gensim_dictionary, docs, model_path, model_name):
    gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize(model_path + '/' + model_name + '.mm', gensim_corpus)
    return gensim_corpus


def make_about_document(model_name, no_below, no_above, num_topics, passes, model_path):
    about_text = 'Model Name: ' + model_name + '\n' + 'Number Below: ' + str(no_below) + '\n' + 'Number Above: ' \
                 + str(no_above) + '\n' + 'Number of Topics: ' + str(num_topics) + '\n' + 'Passes: ' + str(passes) + ''
    with open(model_path + '/' + 'about.txt', 'w') as f:
        f.write(about_text)


def make_gensim_model(model_name, no_below, no_above, num_topics, passes):
    model_name=model_name
    os.mkdir('./models/' + model_name)
    model_path = './models/' + model_name
    # exract docs from json
    with open('./base_corpus/corpus_data.json', 'r') as json_file:
        data = json.load(json_file)
    docs = []
    for key in data.keys():
        doc = data[key]['lemmas']
        docs.append(doc)
    gensim_dictionary = make_gensim_dictionary(docs=docs, no_below=no_below, no_above=no_above,model_path=model_path,
                                               model_name=model_name)
    gensim_corpus = make_gensim_corpus(gensim_dictionary=gensim_dictionary, docs=docs, model_path=model_path,
                                       model_name=model_name)
    lda = models.LdaModel(gensim_corpus, id2word=gensim_dictionary, num_topics=num_topics, passes=passes,
                          random_state=42)
    lda.save(model_path + '/' + model_name + '.model')
    # create about file
    make_about_document(model_name=model_name, no_below=no_below, no_above=no_above, num_topics=num_topics,
                        passes=passes, model_path=model_path)


if __name__ == "__main__":
    model_name = '03'
    no_below = 1000
    no_above = 0.7
    num_topics = 40
    passes = 100
    make_gensim_model(model_name=model_name, no_below=no_below, no_above=no_above, num_topics=num_topics, passes=passes)