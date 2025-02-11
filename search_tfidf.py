import json
import os
import joblib
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import classla
from scipy.sparse import save_npz, load_npz

classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
stop_words = set(stopwords.words('slovene'))

EMBEDDINGS_FILE = 'embeddings/tfidf/embeddings.npz'
VECTORIZER_FILE = 'embeddings/tfidf/vectorizer.pkl'
METADATA_FILE = 'embeddings/tfidf/metadata.json'

class EmbeddingManager:
    """Singleton class to manage embeddings."""
    _instance = None
    _vectorizer = None
    _tfidf_matrix = None
    _metadata = None

    @staticmethod
    def get_instance():
        """Get the singleton instance of the class."""
        if EmbeddingManager._instance is None:
            EmbeddingManager._instance = EmbeddingManager()
        return EmbeddingManager._instance

    def get_vectorizer(self):
        """Retrieve or load the vectorizer."""
        if EmbeddingManager._vectorizer is None:
            EmbeddingManager._vectorizer = joblib.load(VECTORIZER_FILE)
        return EmbeddingManager._vectorizer
    
    def get_tfidf_matrix(self):
        """Retrieve or load tfidf matrix."""
        if EmbeddingManager._tfidf_matrix is None:
            EmbeddingManager._tfidf_matrix = load_npz(EMBEDDINGS_FILE)
        return EmbeddingManager._tfidf_matrix
    
    def get_metadata(self):
        """Retrieve or load tfidf matrix."""
        if EmbeddingManager._metadata is None:
            EmbeddingManager._metadata = np.array(json.load(open(METADATA_FILE, "r")))
        return EmbeddingManager._metadata

    def prepare_data(self):
        with open('ai_act.yaml', 'r') as file:
                data = yaml.safe_load(file)
            
        element_metadata = []
        preprocessed_enote = []

        for d in data['cleni']:
            text = preprocess(d['poglavje']['naslov'] + "\n" + 
                (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') + 
                d['naslov'] + "\n" + 
                d['vsebina'])
            preprocessed_enote.append(text)
            element_metadata.append({
                "id": d['id_elementa'],
                "type": "cleni"
            })
            
        for d in data['tocke']:
            text = preprocess(d['vsebina'])
            preprocessed_enote.append(text)
            element_metadata.append({
                "id": d['id_elementa'],
                "type": "tocke"
            })

        vectorizer = TfidfVectorizer(norm='l2')
        tfidf_matrix = vectorizer.fit_transform(preprocessed_enote)

        if not os.path.exists(os.path.dirname(EMBEDDINGS_FILE)):
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        if not os.path.exists(os.path.dirname(VECTORIZER_FILE)):
            os.makedirs(os.path.dirname(VECTORIZER_FILE), exist_ok=True)
        if not os.path.exists(os.path.dirname(METADATA_FILE)):
            os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

        save_npz(EMBEDDINGS_FILE, tfidf_matrix)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        json.dump(element_metadata, open(METADATA_FILE, "w"))

    def load_embeddings(self):
        """Ensure embeddings are saved in file."""
        if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(VECTORIZER_FILE)):
            print("Getting new embeddings and storing them to file...\n")
            self.prepare_data()
        
        self._vectorizer = joblib.load(VECTORIZER_FILE)
        self._tfidf_matrix = load_npz(EMBEDDINGS_FILE)
        self._metadata = np.array(json.load(open(METADATA_FILE, "r")))


def preprocess(text):
    doc = nlp(text)
    
    tokens = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            lemma = token.words[0].lemma
            if (lemma.isalpha() or lemma.isdigit()) and lemma.lower() not in stop_words:
                tokens.append(lemma.lower())
    
    return ' '.join(tokens)


def search(query, top_n=None):
     # Get the singleton instance and load embeddings
    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    vectorizer = embedding_manager.get_vectorizer()
    tfidf_matrix = embedding_manager.get_tfidf_matrix()
    metadata = embedding_manager.get_metadata()
    
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

    return list(zip([entry['id'] for entry in metadata[top_indices]], similarity_scores[top_indices]))



def get_relevant_results(query="Kdaj začne uredba veljati in se uporabljati?", top_n=None):  
    results = search(query, top_n)

    print("Relevantne enote:")
    for idx, score in results:
        print(f"{idx} s podobnostjo {score}")
