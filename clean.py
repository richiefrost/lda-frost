from textacy.datasets.wikipedia import Wikipedia
from multiprocessing import Pool
import spacy

nlp = spacy.load('en')

stop_titles = set(u'wikipedia category file portal template mediawiki user help book draft'.split())
stopwords = set()
with open("../stopwords.txt") as f:
    for word in f.readlines():
        stopwords.add(unicode(word))

def process_mini_batch(texts):
    result = []
    for text in texts:
        result.append(process_text(text))
    return result

def process_text(text):
    words = []
    text = text.lower()
    tokens = nlp(text)
    for token in tokens[:5]:
        if token.lower_ in stop_titles:
            return []
    for token in tokens[5:]:
        if len(token.lower_) < 4:
            continue
        if token.lower_ in stopwords:
            continue
        if token.like_num:
            continue
        if not token.is_ascii:
            continue
        if token.pos_ in {u'NOUN', u'PROPN'}:
            words.append(token.lemma_)

    return words

pool_size = 32

p = Pool(pool_size)

wp = Wikipedia(lang='en', version='latest')

with open("lemmatized_nouns/output.txt", "w+") as f:
    batch, batch_max = [], 2 ** 14
    
    for text in wp.texts(min_len=300):
        batch.append(text)
        if len(batch) >= batch_max:
            # Returns pool_size number of arrays of roughly (batch_max / pool_size) processed documents (each document represented in array form)
            results = p.map(process_mini_batch, (batch[i::pool_size] for i in range(pool_size)))
            for result in results:
                for entry in result:
                    # Write each document on its own line
                    f.write(' '.join([word.encode('utf-8') for word in entry]) + "\n")
            
            batch = []
