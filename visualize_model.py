from gensim import corpora, models
import pyLDAvis.gensim

model_num = ''
path = './gensim_output/' + model_num + '/'

gensim_dictionary = corpora.Dictionary.load(path + 'gensim_dictionary_' + model_num + '.dict')
gensim_corpus = corpora.MmCorpus(path + 'gensim_corpus_' + model_num + '.mm')

lda_25 = models.ldamodel.LdaModel.load(path + 'lda_25_' + model_num + '.model')

lda_25_viz = pyLDAvis.gensim.prepare(lda_25, gensim_corpus, gensim_dictionary)

pyLDAvis.save_html(lda_25_viz, './visualizations/lda_' + model_num + '.html')