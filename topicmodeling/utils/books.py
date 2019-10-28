from datasets import load_gutenberg
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys
sys.path.append('../matching')
from gaus_marginal_matching import match_local_atoms

def cluster_book(bow_book, embeddings, K):
    idx_active = np.where(bow_book)[0]
    weights = bow_book[idx_active]
    words = embeddings[idx_active]
    kmeans = KMeans(n_clusters=K, n_init=8, max_iter=300, n_jobs=-1).fit(words, sample_weight=weights)
    return kmeans.cluster_centers_

def print_topics(topics, vocab, embeddings, n_top_words=20, meta=None, idx=-1):
    topic_to_word = euclidean_distances(topics, embeddings)
    word_topics = []
    if meta is not None:
        name = meta['title'][idx]
        author = meta['author'][idx]
        print('Topics of ' + name + ' by ' + author)
    for i, topic_dist in enumerate(topic_to_word):
        topic_i = np.array(vocab)[np.argsort(topic_dist)][:n_top_words]
        word_topics.append(topic_i)
        print('Topic {}: {}'.format(i, ' '.join(topic_i)))
    print('\n')
    
    return word_topics
#vocab, embeddings, bow, metadata = load_gutenberg('./data/Gutenberg/',
#                                        './data/glove.6B/glove.6B.50d.txt',
#                                        voc_len=1e6)

#np.save('vocab', vocab)
#np.save('embeddings', embeddings)
#np.save('bow', bow)
#np.save('metadata', metadata)

# vocab = np.load('vocab.npy')
# embeddings = np.load('embeddings.npy')
# bow = np.load('bow.npy', allow_pickle=True)[()]
# metadata = np.load('metadata.npy', allow_pickle=True)[()]
    
book_ids = [0,1,2,3,4]
K = 50

all_topics = []
all_topic_words = []
for b_idx in book_ids:
    book_topics = cluster_book(bow[b_idx].toarray().flatten(), embeddings, K)
    all_topics.append(book_topics)
    topic_words = print_topics(book_topics, vocab, embeddings, meta=metadata, idx=b_idx)
    all_topic_words.append(topic_words)
    print('\n')

matched_topics, popularity, assignments = match_local_atoms(local_atoms=all_topics, sigma=1., sigma0=1., gamma=1., it=100, optimize_hyper=True)

matched_topic_words = print_topics(matched_topics, vocab, embeddings, n_top_words=20, meta=None, idx=-1)

interesting_global_topic = 46
print('Global topic {}: {}\n'.format(interesting_global_topic, ' '.join(matched_topic_words[interesting_global_topic])))

for i, b_idx in enumerate(book_ids):
    if interesting_global_topic in assignments[i]:
        local_idx = assignments[i].index(interesting_global_topic)
        print('Matched topic in ' + metadata['title'][b_idx] + ' by ' + metadata['author'][b_idx])
        print('Topic {}: {}'.format(local_idx, ' '.join(all_topic_words[i][local_idx])))
    else:
        print('Nothing matched in ' + metadata['title'][b_idx] + ' by ' + metadata['author'][b_idx])
    print('\n')
