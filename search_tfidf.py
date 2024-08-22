from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import classla

classla.download('sl')                  

stop_words = set(stopwords.words('slovene'))
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')

def preprocess(text):
    doc = nlp(text)
    
    tokens = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            lemma = token.words[0].lemma
            if (lemma.isalpha() or lemma.isdigit()) and lemma.lower() not in stop_words:
                tokens.append(lemma.lower())
    
    return ' '.join(tokens)


def search(query, tfidf_matrix, vectorizer, top_n=None):
    preprocessed_query = preprocess(query)
    
    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if top_n == None:
        # Take all relevant results
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        # Take only the top n relevant results
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
    
    return top_indices, similarity_scores[top_indices]


def prepare_data():
    with open('ai_act.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    cleni = [f"{d['id_elementa']}" for d in data['cleni']]
    tocke = [f"{d['id_elementa']}" for d in data['tocke']]
    enote = cleni + tocke

    preprocessed_cleni = [
        preprocess(
            d['poglavje']['naslov'] + "\n" + 
            (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') + 
            d['naslov'] + "\n" + 
            d['vsebina']
        ) for d in data['cleni']]
    
    preprocessed_tocke = [
        preprocess(d['vsebina']) for d in data['tocke']
    ]

    preprocessed_enote = preprocessed_cleni + preprocessed_tocke

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_enote)

    return enote, tfidf_matrix, vectorizer


def get_relevant_results(query="Kdaj začne uredba veljati in se uporabljati?", top_n=None):
    enote, tfidf_matrix, vectorizer = prepare_data()
    
    top_indices, scores = search(query, tfidf_matrix, vectorizer, top_n)

    print("Relevantne enote:")
    for idx, score in zip(top_indices, scores):
        print(f"{enote[idx]} s podobnostjo {score}")