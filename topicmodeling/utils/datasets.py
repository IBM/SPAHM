import collections
import os
import numpy as np

from scipy.sparse import csr_matrix
from scipy.io import savemat

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

from word_utils import change_embeddings


def load_files(folder, num_docs=None):
    filenames = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            filenames.append(os.path.join(path, name))
    if num_docs is not None:
        filenames = np.random.choice(filenames, size=num_docs, replace=False)

    data = []
    metadata = collections.defaultdict(list)
    for filename in filenames:
        fname = os.path.basename(filename)

        title = os.path.splitext(fname)[0]
        author = title.split('_')[0]
        title = title.split('_')[-1]
        
        try:
            print('Opening and reading {}'.format(fname))
            
            with open(filename, encoding='utf-8') as f:
                data.append(f.read())
                
            metadata['author'].append(author)
            metadata['title'].append(title)
        except:
            pass
    return data, metadata


def load_gutenberg(folder, embed_path, stem=False, voc_len=15000):
    data, metadata = load_files(folder, num_docs=None)
    print('Read {} documents'.format(len(data)))

    counter = CountVectorizer(token_pattern=r'\b[a-zA-Z]+\b')
    bow_data = counter.fit_transform(data)

    # Remove stop words and short words from vocabulary
    stop_words = set(stopwords.words('english'))
    # Additional stop words based on preliminary results
    new_stop_words = ['thou', 'thee', 'thy', 'one', 'two', 'sir', 'would', 'could',
                      'should', 'shall', 'may', 'upon', 'also', 'said', 'mrs', 'miss']
    stop_words.update(new_stop_words)
    vocab = counter.get_feature_names()

    vocab, embed_vocab, bow_data = change_embeddings(vocab,
                                                     bow_data,
                                                     embed_path)
    vocab_idx = []
    for i, w in enumerate(vocab):
        if len(w) > 2 and w not in stop_words:
            vocab_idx.append(i)
    vocab = [vocab[i] for i in vocab_idx]
    voc_len = min(voc_len, len(vocab))
    bow_data = bow_data[:, vocab_idx]

    # Loaded all data
    sum_words = bow_data.sum(axis=0)
    words_freq = [(word, sum_words[0, idx], idx) for idx, word in
                  enumerate(vocab)]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print('Vocabulary size: {}'.format(len(vocab)))
    print('Most common words: {}'.format(words_freq[:10]))
    print('Least common words: {}'.format(words_freq[-10:]))

    print('Truncating vocabulary to {} words'.format(voc_len))
    vocab = [w for (w, cnt, idx) in words_freq[:voc_len]]
    vocab_idx = [idx for (w, cnt, idx) in words_freq[:voc_len]]
    embed_vocab = {w: embed_vocab[w] for w in vocab}
    bow_data = bow_data[:, vocab_idx]
    embeddings = np.array([embed_vocab[w] for w in vocab])

    filename = '_'.join(['gutenberg', str(voc_len), 'reduced']) + '.mat'
    savemat(filename, {'X': embeddings,
                       'BOW_X': bow_data,
                       'words': vocab,
                       'authors': metadata['author'],
                       'titles': metadata['title']})
    print('Reduced vocabulary to {} words.'.format(len(vocab)))

    if not stem:
        return vocab, embeddings, bow_data, metadata
    # Stemming with Snowball stemmer
    stemmer = SnowballStemmer('english')
    stemmed_vocab = set([])
    stemmed_idx = collections.defaultdict(list)
    stemmed_embed_vocab = {}
    for i, word in enumerate(vocab):
        stemmed = stemmer.stem(word)
        stemmed_vocab.add(stemmed)
        stemmed_idx[stemmed].append(i)
        if stemmed not in stemmed_embed_vocab:
            stemmed_embed_vocab[stemmed] = np.array(embed_vocab[word])
        else:
            n = float(len(stemmed_idx))
            stemmed_embed_vocab[stemmed] = ((n - 1) / n * stemmed_embed_vocab[stemmed]
                                            + 1 / n * np.array(embed_vocab[word]))

    stemmed_vocab = list(stemmed_vocab)
    print('Reduced vocabulary to {} words.'.format(len(stemmed_vocab)))

    stemmed_bow_data = csr_matrix((bow_data.shape[0], len(stemmed_vocab)),
                                  dtype=np.int8)
    for i, stemmed in enumerate(stemmed_vocab):
        print('Processing word {}/{}'.format(i, len(stemmed_vocab)))
        for j in stemmed_idx[stemmed]:
            stemmed_bow_data[:, i] = stemmed_bow_data[:, i] + bow_data[:, j]
    stemmed_embeddings = np.array([stemmed_embed_vocab[w] for w in stemmed_vocab])

    # Save this to a .mat file for later
    savemat('gutenberg_stemmed.mat', {'X': stemmed_embeddings,
                                      'BOW_X': stemmed_bow_data,
                                      'words': stemmed_vocab})
    return stemmed_vocab, stemmed_embeddings, stemmed_bow_data
