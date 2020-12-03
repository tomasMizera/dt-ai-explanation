#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import time


# ---
# ### Load data

# In[2]:


def load_polarity(path='../data/rt-polaritydata'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels


# In[3]:


nlp = spacy.load('en_core_web_lg')


# In[4]:


data, labels = load_polarity()


# In[5]:


train, test, train_labels, test_labels =     sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels =     sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)


# In[6]:


train[:4]


# In[7]:


train_labels[:4]


# Convert labels to np array

# In[8]:


train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)


# In[9]:


train_labels[:4]


# ---
# ### Text preprocessing

# In[10]:


vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)


# In[11]:


train_vectors = vectorizer.transform(train)


# In[12]:


test_vectors = vectorizer.transform(test)


# In[13]:


val_vectors = vectorizer.transform(val)


# ---

# ### Training

# In[14]:


model = sklearn.linear_model.LogisticRegression( max_iter=1000 )
model.fit(train_vectors, train_labels)


# In[15]:


# predict
preds = model.predict(val_vectors)


# In[16]:


print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))


# ---

# ### Explanation in iterations

# ---

# In[90]:


# define a decorator to log execusion time
# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d

def timeit(method):
    def timed(*args, **kw):
        timed.calls += 1
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
#         if 'log_time' in kw:
#             name = kw.get('log_name', method.__name__.upper())
#             kw['log_time'][name] = int((te - ts) * 1000)
#         else:
#             print('%r  %2.2f ms' % \
#                   (method.__name__, (te - ts) * 1000))
        timed.time_taken += (te - ts) * 1000
        return result
    timed.calls = 0
    timed.time_taken = 0
    return timed


# In[91]:


# this is the requested function by Anchors!

@timeit
def predict_text(text):
    return model.predict(vectorizer.transform(text))


# In[57]:


# build explanator
explanator = anchor_text.AnchorText(nlp, ["negative", "positive"], use_unk_distribution=False)


# In[55]:


predict_text(["Good film"])


# In[85]:


explain_sample = train[:30]


# In[25]:


explain_sample[:2]


# In[63]:


explain_sample[0]


# In[92]:


explanation = explanator.explain_instance("rare birds", predict_text, threshold=0.95, verbose=False, use_proba=True)


# In[93]:


explanation.coverage()


# In[94]:


def exp_ratio(explanation):
    cov = explanation.coverage()
    prec = explanation.precision()
    return (prec - cov)/max(cov, prec)


# In[84]:


exp_ratio(explanation)


# ----

# ### Iteration

# ---

# In[104]:


data = explain_sample
expl = explanator


# In[103]:


# testing purposes
# data = ["good worst", "bad", "worst", "best"]


# In[108]:


data = list(map(lambda x: str(x), data))


# In[ ]:


for sample in data:
    print("Processing: " + sample)
    explanation = expl.explain_instance(sample, predict_text, threshold=0.95, verbose=False, use_proba=True)
    print('Took:  %2.2f ms' %                   (predict_text.time_taken))
    print(f'Called {predict_text.calls} times')
    predict_text.calls = 0
    predict_text.time_taken = 0
    
    # process explanation
    print(f'Ratio: {exp_ratio(explanation)}')
    print(f'Cov: {explanation.coverage()}, Prec: {explanation.precision()} \n ----- ')

