from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction

def vectorize_messages(cleaned_mex, model, tf_idf=False):
    """
    Calculates mathematical average of the word vector representations
    of all the words in each sentence
    """
    sent2vec = []
    tokenized= [row.split() for row in cleaned_mex]
    if tf_idf:
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(tokenized)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        sent2vec = np.array([
            np.mean([model.wv[w] * word2weight[w]
                     for w in words if w in model.wv] or
                    [np.zeros(model.vector_size)], axis=0)
            for words in tokenized
        ])
    else:
        for sent in tokenized:
            sent_vec = np.average([model.wv[w] if w in model.wv else np.zeros((model.vector_size,), dtype=np.float32)
                               for w in sent], 0)
            sent2vec.append(np.zeros((model.vector_size,), dtype=np.float32) if np.isnan(np.sum(sent_vec)) else sent_vec)
        sent2vec = np.array(sent2vec)
    return sent2vec

def reduce_dimensions(model):
    """
    TSNE reduction of dimensions (2D)
    """   
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def reduce_sent_dimensions(df):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for index, row in df.iterrows():
        vectors.append(row['vectors_sent'])
        labels.append(row['cleaned_strings'])

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_matplotlib(x_vals, y_vals, labels,figsize,npoints=100,save=False, title='w2v_visualization.png'):
    
    import matplotlib.pyplot as plt
    import random
    random.seed(0)
    plt.figure(figsize=figsize)
    plt.scatter(x_vals, y_vals)
    plt.title(title)
    
    import mplcursors
    mplcursors.cursor(highlight=True, multiple=True).connect(
        "add", lambda sel: sel.annotation.set_text(
              labels[sel.target.index]
    ))

    # Label randomly subsampled npoints data points
    
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, npoints)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    if(save):
        plt.savefig(title)
    else:
        plt.show()

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print delta loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.loss_vec=[]

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.loss_vec.append(loss)
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.loss_vec.append(loss-self.loss_previous_step)

        self.epoch += 1
        self.loss_previous_step = loss

        
