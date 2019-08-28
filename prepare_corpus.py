# imports
import os
import xml.etree.ElementTree as ET
import spacy
from gensim import corpora
import json


# set up nlp for later use
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
custom_stops = {'professor', 'cambridge', 'author', 'chapter', 'volume', 'essay', 'volumne', 'publish', 'page', 'paper',
                'article', 'introduction', 'edition', 'cit', 'op', 'note', 'oxford', 'ed', 'routledge', 'york', 'eds',
                'vol', 'blackwell', 'subscription', 'journal', 'department', 'pp'}
stop_words = stop_words.union(custom_stops)


def extract_tokens(ngrams):
    article_words = []
    text = ngrams.rstrip()
    text_list = text.split('\n')
    for item in text_list:
        li = item.split('\t')
        if len(li[0]) < 2:  # data cleanup: eliminate ocr noise
            continue
        word = li[0] + ' '
        count = int(li[1])
        word_string = word * count
        word_string = word_string.rstrip()  # remove extra space at the end
        word_list = word_string.split(' ')
        for word in word_list:
            article_words.append(word)
    token_string = ' '.join(article_words)
    return token_string


def process_text(string):
    doc = nlp(string)
    tokens = [token for token in doc]
    lemmas_alpha = [token.lemma_ for token in tokens if token.is_alpha]
    lemmas_no_pron = [lemma for lemma in lemmas_alpha if lemma != '-PRON-']
    lemmas_final = [lemma for lemma in lemmas_no_pron if lemma not in stop_words]
    return lemmas_final


ngram1_path = './jstor_data/ngram1/'
txt_files = sorted(os.listdir(ngram1_path))[:5]

metadata_path = './jstor_data/metadata/'
xml_files = sorted(os.listdir(metadata_path))[:5]


docs = []
corpus_dict = {}
i = 0
unwanted_titles = ['volume information', 'front matter', 'back matter']

for xml_file, txt_file in zip(xml_files, txt_files):
    # parse xml file
    tree = ET.parse(metadata_path + xml_file)
    root = tree.getroot()
    # get title
    title = root.find('front/article-meta/title-group/article-title')
    try:
        title = title.text.lower()
    except AttributeError:
        title = 'book review'  # book reviews do not have titles
    # filter out issue infomration
    if title in unwanted_titles:
        continue
    year = root.find('front/article-meta/pub-date/year')
    journal = root.find('front/journal-meta/journal-title-group/journal-title')
    doi = root.find('front/article-meta/article-id')
    article_dict = {}
    article_dict['doi'] = doi.text.replace('/', '-')
    article_dict['title'] = title
    article_dict['journal'] = journal.text
    article_dict['year'] = year.text

    key = 'doc_' + str(i)
    corpus_dict[key] = article_dict

    # process txt file
    with open(file=ngram1_path + txt_file, encoding="utf8", mode="r") as f:
        ngrams = f.read()
        article_string = extract_tokens(ngrams)
        lemmas = process_text(article_string)
        docs.append(lemmas)
        print('Finished document', str(i))  # display progress
        i += 1

# add output for metadatafile
with open('./gensim_output/corpus_data.json', 'w') as outfile:
    json.dump(corpus_dict, outfile)


# # gensim dictionary
# gensim_dictionary = corpora.Dictionary(docs)
# gensim_dictionary.filter_extremes(no_below=1000, no_above=0.7)
# gensim_dictionary.save('./gensim_output/gensim_dictionary.dict')
#
# # gensim corpus
# gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in docs]
# corpora.MmCorpus.serialize('./gensim_output/gensim_corpus.mm', gensim_corpus)
