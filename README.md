# jstor-topic-model
Create and visualize topic models from jstor ngrams with gensim

### Directory Structure

The project currently assumes the following directories:
* `gensim_output` - this directory contains all the gensim output including the corpus
dictionary, the corpus, and any models and their assosciated files.
* `jstor_data` - this directory contains two subdirectories, `metadata` and `ngram1`, which
 contain the data provided by JSTOR's [Data for Research](https://www.jstor.org/dfr/about/creating-datasets).
    * `metadata` - this subdirectory contains the xml metadata
    * `ngram1` - this subdirectory contains ngram1 data
* `visualizations` - this directory contains html files for visualizing the topic models

