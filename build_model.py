from gensim import corpora, models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_num = ''
path = './gensim_output/' + model_num + '/'

gensim_dictionary = corpora.Dictionary.load(path + 'gensim_dictionary_' + model_num + '.dict')


gensim_corpus = corpora.MmCorpus(path + 'gensim_corpus_' + model_num + '.mm')

lda_25 = models.LdaModel(gensim_corpus, id2word=gensim_dictionary, num_topics=25, passes=100, random_state=42)
lda_25.save(path + 'lda_25_' + model_num + '.model')