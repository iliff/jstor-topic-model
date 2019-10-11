# imports
import os
from collections import OrderedDict
import xml.etree.ElementTree as ET
import spacy
import json


def extract_tokens(ngrams):
    """Extract and count tokens from JSTOR ngram data"""
    article_words = []
    text = ngrams.rstrip()
    text_list = text.split('\n')
    for item in text_list:
        word_count = item.split('\t')
        if len(word_count[0]) < 3:  # data cleanup: eliminate ocr noise and most roman numerals
            continue
        word = word_count[0] + ' '
        count = int(word_count[1])
        word_string = word * count
        word_string = word_string.rstrip()  # remove extra space at the end
        word_list = word_string.split(' ')
        for word in word_list:
            article_words.append(word)
    token_string = ' '.join(article_words)
    return token_string


def process_text(string, custom_stops={}):
    """Process text using SpaCy"""
    nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    roman_numerals = {'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ', 'ⅴ', 'ⅵ', 'ⅶ', 'ⅷ', 'ⅸ', 'ⅹ', 'ⅺ', 'ⅻ', 'ⅹⅲ', 'ⅹⅳ', 'ⅹⅴ', 'ⅹⅵ', 'ⅹⅶ',
                      'ⅹⅷ', 'ⅹⅸ', 'ⅹⅹ', 'ⅹⅺ', 'ⅹⅻ', 'ⅹⅹⅲ', 'ⅹⅹⅳ', 'ⅹⅹⅴ', 'ⅹⅹⅵ', 'ⅹⅹⅶ', 'ⅹⅹⅷ', 'ⅹⅹⅸ', 'ⅹⅹⅹ', 'ⅹⅹⅺ',
                      'ⅹⅹⅻ', 'ⅹⅹⅹⅲ', 'ⅹⅹⅹⅳ', 'ⅹⅹⅹⅴ', 'ⅹⅹⅹⅵ', 'ⅹⅹⅹⅶ', 'ⅹⅹⅹⅷ', 'ⅹⅹⅹⅸ', 'ⅹⅼ', 'ⅹⅼⅰ', 'ⅹⅼⅱ', 'ⅹⅼⅲ', 'ⅹⅼⅳ',
                      'ⅹⅼⅴ', 'ⅹⅼⅵ', 'ⅹⅼⅶ', 'ⅹⅼⅷ', 'ⅹⅼⅸ', 'ⅼ'}
    stop_words = stop_words.union(roman_numerals)
    stop_words = stop_words.union(custom_stops)
    doc = nlp(string)
    tokens = [token for token in doc]
    lemmas_alpha = [token.lemma_ for token in tokens if token.is_alpha]
    lemmas_no_pron = [lemma for lemma in lemmas_alpha if lemma != '-PRON-']
    lemmas_final = [lemma for lemma in lemmas_no_pron if lemma not in stop_words]
    return lemmas_final


def prepare_corpus(custom_stops={}):
    os.mkdir('./base_corpus/')
    ngram1_path = './jstor_data/ngram1/'
    txt_files = sorted(os.listdir(ngram1_path))

    metadata_path = './jstor_data/metadata/'
    xml_files = sorted(os.listdir(metadata_path))

    docs = []
    corpus_metadata = OrderedDict()
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
        # process txt file
        with open(file=ngram1_path + txt_file, encoding="utf8", mode="r") as f:
            ngrams = f.read()
            article_string = extract_tokens(ngrams)
            lemmas = process_text(article_string, custom_stops=custom_stops)
            article_dict['lemmas'] = lemmas
            docs.append(lemmas)
        key = 'doc_' + str(i)
        corpus_metadata[key] = article_dict
        if i % 100 == 0:
            print('Finished document', str(i), '/', str(len(txt_files)))  # display progress
        i += 1
# add output for metadata file
    with open('./base_corpus/corpus_data.json', 'w') as outfile:
        json.dump(corpus_metadata, outfile)


if __name__ == "__main__":

    custom_stop_words = {'address', 'article', 'association', 'author', 'blackwell', 'book', 'cambridge', 'chapter',
                         'chicago', 'cit', 'cloth', 'co', 'college', 'committee', 'conference', 'david', 'de',
                         'department', 'der', 'des', 'dr', 'ed', 'edition', 'eds', 'essay', 'follow', 'introduction',
                         'john', 'journal', 'les', 'london', 'meeting', 'mit', 'mr', 'note', 'op', 'oxford', 'page',
                         'paper', 'paul', 'pp',  'president', 'press', 'prof', 'professor', 'publish', 'richard',
                         'robert', 'routledge', 'society', 'subscription', 'uk', 'und', 'vol', 'volume', 'volumne',
                         'von', 'william', 'york'}

    prepare_corpus(custom_stops=custom_stop_words)