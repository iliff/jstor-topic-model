from gensim import corpora, models
import pyLDAvis.gensim


def visualize_model(model_name):

    path = './models/' + model_name + '/'
    gensim_dictionary = corpora.Dictionary.load(path + model_name + '.dict')
    gensim_corpus = corpora.MmCorpus(path + model_name + '.mm')
    model = models.ldamodel.LdaModel.load(path + model_name + '.model')
    model_viz = pyLDAvis.gensim.prepare(model, gensim_corpus, gensim_dictionary)
    pyLDAvis.save_html(model_viz, path + model_name + '.html')

if __name__ == '__main__':

    visualize_model(model_name='03')