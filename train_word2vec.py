

from gensim.models import word2vec

if __name__ == '__main__':

    #corpus_model = create_corpus('location_corpus.txt', 'location_corpus.model')
    #corpus_model = Corpus.load('corpus.model')
    #glove = train_glove(corpus_model, 'location_glove.model')

    with open('location_corpus.txt', 'r') as f:
        sentences = [ line.rstrip('\n').split(' , ') for line in f ]
    countries = [ s[-1] for s in sentences ]
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=0, workers=4)
