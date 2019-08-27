from gensim import corpora, models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = './gensim_output/'

gensim_dictionary = corpora.Dictionary.load(path + 'gensim_dictionary.dict')


gensim_corpus = corpora.MmCorpus(path + 'gensim_corpus.mm')

lda_25 = models.LdaModel(gensim_corpus, id2word=gensim_dictionary, num_topics=25, passes=100, random_state=42)
lda_25.save(path + 'lda_25.model')