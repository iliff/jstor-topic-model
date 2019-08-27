from gensim import corpora, models
import pyLDAvis.gensim

path = './gensim_output/'

gensim_dictionary = corpora.Dictionary.load(path + 'gensim_dictionary.dict')
gensim_corpus = corpora.MmCorpus(path + 'gensim_corpus.mm')

lda_25 = models.ldamodel.LdaModel.load(path + 'lda_25.model')

lda_25_viz = pyLDAvis.gensim.prepare(lda_25, gensim_corpus, gensim_dictionary)

pyLDAvis.save_html(lda_25_viz, './visualizations/lda.html')