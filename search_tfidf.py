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

EMBEDDINGS_FILE = "tfidf_embeddings/{}/embeddings.npz"
VECTORIZER_FILE = "tfidf_embeddings/{}/vectorizer.pkl"
METADATA_FILE = "tfidf_embeddings/{}/metadata.json"


class EmbeddingManager:
    """Singleton class to manage embeddings."""
    _instance = None
    _vectorizers = {}
    _tfidf_matrices = {}
    _metadata = {}

    @staticmethod
    def get_instance():
        """Get the singleton instance of the class."""
        if EmbeddingManager._instance is None:
            EmbeddingManager._instance = EmbeddingManager()
        return EmbeddingManager._instance

    def get_vectorizer(self, unit_conditions):
        """Retrieve or load the vectorizer."""
        unit = self.get_prefix_from_keys(unit_conditions)
        if unit not in self._vectorizers or self._vectorizers[unit] is None:
            self._vectorizers[unit] = joblib.load(VECTORIZER_FILE.format(unit))
        return self._vectorizers[unit]

    def get_tfidf_matrix(self, unit_conditions):
        """Retrieve or load tfidf matrix."""
        unit = self.get_prefix_from_keys(unit_conditions)
        if unit not in self._tfidf_matrices or self._tfidf_matrices[unit] is None:
            self._tfidf_matrices[unit] = load_npz(EMBEDDINGS_FILE.format(unit))
        return self._tfidf_matrices[unit]

    def get_metadata(self, unit_conditions):
        """Retrieve or load metadata."""
        unit = self.get_prefix_from_keys(unit_conditions)
        if unit not in self._metadata or self._metadata[unit] is None:
            self._metadata[unit] = np.array(json.load(open(METADATA_FILE.format(unit), "r")))
        return self._metadata[unit]

    @staticmethod
    def process_section(data, key):
        """Processes either 'cleni' or 'tocke'."""
        metadata = []
        preprocessed_texts = []

        for d in data[key]:
            if key == "cleni":
                text = preprocess(
                    d['poglavje']['naslov'] + "\n" +
                    (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') +
                    d['naslov'] + "\n" +
                    d['vsebina'])
            else:  # "tocke"
                text = preprocess(d['vsebina'])

            preprocessed_texts.append(text)
            metadata.append({"id": d['id_elementa'], "type": key})

        return {"metadata": metadata, "preprocessed_embeddings": preprocessed_texts}

    @staticmethod
    def save_data(vectorizer, tfidf_matrix, metadata, prefix):
        """Saves vectorizer, TF-IDF matrix, and metadata with the given prefix."""
        if not os.path.exists(os.path.dirname(EMBEDDINGS_FILE.format(prefix))):
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE.format(prefix)), exist_ok=True)
        if not os.path.exists(os.path.dirname(VECTORIZER_FILE.format(prefix))):
            os.makedirs(os.path.dirname(VECTORIZER_FILE.format(prefix)), exist_ok=True)
        if not os.path.exists(os.path.dirname(METADATA_FILE.format(prefix))):
            os.makedirs(os.path.dirname(METADATA_FILE.format(prefix)), exist_ok=True)

        save_npz(EMBEDDINGS_FILE.format(prefix), tfidf_matrix)
        joblib.dump(vectorizer, VECTORIZER_FILE.format(prefix))
        json.dump(metadata, open(METADATA_FILE.format(prefix), "w"))

    @staticmethod
    def get_prefix_from_keys(key: list[str]):
        if key is None or set(key) == {"cleni", "tocke"}:
            return "cleni_tocke"
        elif set(key) == {"cleni"}:
            return "cleni"
        elif set(key) == {"tocke"}:
            return "tocke"
        else:
            raise ValueError(f"Invalid key: {key}. Expected None, ['cleni', 'tocke'], ['cleni'], or ['tocke'].")

    def prepare_data(self):
        with open('ai_act.yaml', 'r') as file:
            data = yaml.safe_load(file)

        datasets = {}
        for key in ["cleni", "tocke"]:
            datasets[key] = self.process_section(data, key)

        all_preprocessed_embeddings = []
        all_metadata = []

        for key, values in datasets.items():
            all_preprocessed_embeddings += values["preprocessed_embeddings"]
            all_metadata += values["metadata"]

            vectorizer = TfidfVectorizer(norm='l2')
            tfidf_matrix = vectorizer.fit_transform(values['preprocessed_embeddings'])

            self.save_data(vectorizer, tfidf_matrix, values['metadata'], key)

        vectorizer = TfidfVectorizer(norm='l2')
        tfidf_matrix = vectorizer.fit_transform(all_preprocessed_embeddings)

        self.save_data(vectorizer, tfidf_matrix, all_metadata, "cleni_tocke")

    def load_data(self, prefix):
        self._vectorizers[prefix] = joblib.load(VECTORIZER_FILE.format(prefix))
        self._tfidf_matrices[prefix] = load_npz(EMBEDDINGS_FILE.format(prefix))
        self._metadata[prefix] = np.array(json.load(open(METADATA_FILE.format(prefix), "r")))

    def load_embeddings(self):
        """Ensure embeddings are saved in file."""
        files_exist = True & (os.path.exists(EMBEDDINGS_FILE.format("cleni_tocke")) and os.path.exists(
            VECTORIZER_FILE.format("cleni_tocke")) and os.path.exists(METADATA_FILE.format("cleni_tocke")))
        for unit_prefix in ["cleni", "tocke"]:
            files_exist &= (os.path.exists(EMBEDDINGS_FILE.format(unit_prefix)) and os.path.exists(
                VECTORIZER_FILE.format(unit_prefix)) and os.path.exists(METADATA_FILE.format(unit_prefix)))

        if not files_exist:
            print("Getting new embeddings and storing them to file...\n")
            self.prepare_data()

        self.load_data("cleni")
        self.load_data("tocke")
        self.load_data("cleni_tocke")


def preprocess(text):
    doc = nlp(text)

    tokens = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            lemma = token.words[0].lemma
            if (lemma.isalpha() or lemma.isdigit()) and lemma.lower() not in stop_words:
                tokens.append(lemma.lower())

    return ' '.join(tokens)


def search(query, top_n=None, unit_conditions: list[str] = None):
    if unit_conditions is None:
        unit_conditions = list(['cleni', 'tocke'])

    # Get the singleton instance and load embeddings
    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    vectorizer = embedding_manager.get_vectorizer(unit_conditions)
    tfidf_matrix = embedding_manager.get_tfidf_matrix(unit_conditions)
    metadata = embedding_manager.get_metadata(unit_conditions)

    preprocessed_query = preprocess(query)

    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if top_n is None:
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
