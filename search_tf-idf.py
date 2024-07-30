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
            if lemma.isalpha() and lemma.lower() not in stop_words:
                tokens.append(lemma.lower())
    
    return ' '.join(tokens)


# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     tokens = [word for word in tokens if word not in stop_words]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     return ' '.join(tokens)


def search(query, tfidf_matrix, vectorizer, top_n=10):
    preprocessed_query = preprocess(query)
    
    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
    
    return top_indices, similarity_scores[top_indices]


if __name__ == '__main__':
    with open('ai_act.yaml', 'r') as file:
        data = yaml.safe_load(file)

    clen_numbers = [d['clen'] for d in data['cleni']]
    preprocessed_cleni = [
        preprocess(
            d['poglavje']['naslov'] + "\n" + 
            (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') + 
            d['naslov'] + "\n" + 
            d['vsebina']
        ) for d in data['cleni']]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_cleni)

    query = "Kakšne so lahko globe?"
    top_indices, scores = search(query, tfidf_matrix, vectorizer)

    for idx, score in zip(top_indices, scores):
        clen_number = clen_numbers[idx]
        print(f"Člen {clen_number} with score {score}")