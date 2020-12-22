#!/usr/bin/env python
# coding: utf-8

# Prototype of model explanation via Anchors with help of extractive summary
# ---

# In[1]:

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import sys
import time
from anchor import anchor_text
import spacy



from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers import _summarizer

import os

tfds.disable_progress_bar()


# In[2]:


now = time.strftime("%Y-%m-%d_%H:%M")


# In[3]:




logging.basicConfig(
    level=logging.DEBUG, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename=f'../data/logs/v1-{now}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

l = logging.getLogger('prototype')
l.critical("Logging prototype v1 with TF model")


# ---

# In[4]:


def _load_model():
    """
    Define a function that loads a model to be explained and returns its instance
    """
    
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset.element_spec
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    VOCAB_SIZE=1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    model.load_weights('../raw-data/lstm-model-v1')
    
    return model    


# In[5]:


model = _load_model()
l.info("Model loaded")


# In[6]:


model.predict(["hahahahahahahahahaha this is the most funny film I have ever seen"])


# ---

# Explanation
# ---

# ### 1. Preparation

# In[7]:





# In[8]:


LANGUAGE = "english"
SENTENCES_COUNT = 6
nlp = spacy.load("en_core_web_lg")


# In[97]:


in_file = "../data/reviews/review-top.txt"
in_files = ["review-top.txt", "review-med.txt", "review-low.txt"]
in_files_path = list(map(lambda x: os.path.join("../data/reviews", x), in_files))

test_input = "hahahahahahahahahaha this is the most funny film I have ever seen"


# In[10]:


file_data = None

with open(in_file, 'r') as f:
    file_data = f.read()


# In[11]:


parser = PlaintextParser.from_file(in_file, Tokenizer(LANGUAGE))

summarizer = TextRankSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = get_stop_words('slovak')

helper = _summarizer.AbstractSummarizer()


# In[36]:


explanator = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)


# In[13]:


# define a decorator to log execusion time
# inspired by https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d

def timeit(method):
    def timed(*args, **kw):
        timed.calls += 1
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        timed.time_taken += (te - ts) * 1000
        return result
    timed.calls = 0
    timed.time_taken = 0
    return timed


# ---

# ### 2. Execution
# 
# > Example for positive review

# In[14]:


l.info("Starting an algorithm")


# **2.1 Model prediction**

# In[15]:


@timeit
def _predict(_input):
    """
    Define a function that takes in a model input (1d array) and returns a prediction (1d array)
    """
    
    return model.predict(_input)[0]


# In[16]:


_predict([test_input])


# **2.2 Instance explanation**

# In[37]:


def _explain_instance(file):
    explanation = explanator.explain_instance(file, _predict, threshold=0.95, verbose=False, use_proba=True)
    l.info('Took:  %2.2f ms' %                   (_predict.time_taken))
    l.info(f'Called {_predict.calls} times')
    l.info(' AND '.join(explanation.names()))
    l.info(f'Precision: {explanation.precision()}')
    l.info(f'Coverage: {explanation.coverage()}')
    _predict.calls = 0
    _predict.time_taken = 0
    
    return explanation


# In[45]:


expl = _explain_instance("not great hm")


# **2.3 Calculate importance**

# In[103]:


def _calc_importance(explanation):
    cov = explanation.coverage()
    prec = explanation.precision()
    
    if max(cov, prec) == 0:
        l.debug("Importance denominator is 0, thus importancy is 1")
        return 1
    
    # updated relative change
    importancy = 1 + abs((prec - cov)/max(cov, prec))
    
    l.info(f'Importance: {importancy}')
    return importancy


# In[104]:


_calc_importance(expl)


# **2.4 Summarization of document**

# In[81]:


# example how to override sumy
helper._get_best_sentences(parser.document.sentences, 2, summarizer.rate_sentences(parser.document))


# In[85]:


def _summarize_doc_basic():
    return summarizer(parser.document, SENTENCES_COUNT)


def _summarize_doc_custom(explanation, importance):
    rates = summarizer.rate_sentences(parser.document)
    
    for sentence in rates.keys():
        # iterate over sentences and if any word from anchor matches a word in sentence, bigger sentences importancy
        if any([anchor_word in str(sentence) for anchor_word in explanation.names()]):
            l.debug("Changing importancy of sentence: " + str(sentence) + " from: " + str(rates[sentence]))
            rates[sentence] = rates[sentence] * importance
            l.debug("to: " + str(rates[sentence]))
    
    resulting_summary = helper._get_best_sentences(parser.document.sentences, SENTENCES_COUNT, rates)
    
    l.info("Resulting summary:")
    l.info(str(resulting_summary))
    
    return resulting_summary


# **2.5 Predict summarized text**

# In[115]:


# helper function to join summary
def _get_data_from_summary(summary):
    return ' '.join(list(map(lambda sentence: str(sentence), summary)))


# **Running all**

# In[98]:


in_files_path_test = list(map(lambda x: x + "-test", in_files_path))
in_files_path_test_setup = list(map(lambda x: x + "-test-setup", in_files_path))


# In[116]:


for file in in_files_path:
    l.info("Processing: " + file)
    
    file_data = None
    with open(file, 'r') as f:
        file_data = f.read()
        
    l.info(f'Model decision on instance: {_predict([file_data])}')
    
    explanation = _explain_instance(file_data)
    
    # process explanation
    importance = _calc_importance(explanation)
    
    summary = _summarize_doc_custom(explanation, importance)
    
    summarized_data = _get_data_from_summary(summary)
    l.info(f'Model decision on summarized instance: {_predict([summarized_data])}')
    
    l.info('Done processing: ' + file + '\n ===== ')

