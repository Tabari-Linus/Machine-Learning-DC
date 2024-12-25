import nltk
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
import numpy as np
import random 
import pickle
from collections import Counter
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000




def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2 = [w for w in w_counts if 1000 > w_counts[w] > 50]
    return l2


def sample_handling(sample, lexicon, classification):
    lexicon_index = {word: idx for idx, word in enumerate(lexicon)}
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon_index:
                    index_value = lexicon_index[word]
                    features[index_value] += 1
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    
    random.seed(42)
    random.shuffle(features)

    # Separate features (X) and labels (y)
    X = [feature[0] for feature in features]
    y = [feature[1] for feature in features]

    X = np.array(X)
    y = np.array(y)

    testing_size = int(test_size * len(X))

    train_x = X[:-testing_size]
    train_y = y[:-testing_size]
    test_x = X[-testing_size:]
    test_y = y[-testing_size:]

    return train_x, train_y, test_x, test_y



if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(
        'Data/pos.txt', 'Data/neg.txt'
    )
    with open('Data/sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
