import yaml
import search_tfidf as s_tfidf
import search_sBERT as s_sbert


def get_tfidf_avg_rank():
    avg_rank = 0

    enote, tfidf_matrix, vectorizer = s_tfidf.prepare_data()

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        top_indices, _ = s_tfidf.search(query, tfidf_matrix, vectorizer)

        target_unit_rank = 1
        for idx in top_indices:
            if enote[idx] == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank

    return avg_rank / len(test_data)


def get_sbert_avg_rank():
    avg_rank = 0

    enote, preprocessed_enote_embeddings = s_sbert.prepare_data()
    
    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        top_indices, _ = s_sbert.search(query, preprocessed_enote_embeddings)

        target_unit_rank = 1
        for idx in top_indices:
            if enote[idx] == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank

    return avg_rank / len(test_data)


if __name__ == '__main__':
    with open('test_set.yaml', 'r') as file:
        test_data = yaml.safe_load(file)    

    tfidf_avg_rank = get_tfidf_avg_rank()

    with open('test_set.yaml', 'r') as file:
        test_data = yaml.safe_load(file)    

    sbert_avg_rank = get_sbert_avg_rank()

    print(f"Povprečen rank ciljne enote s TF-IDF: {tfidf_avg_rank:.2f}")
    print(f"Povprečen rank ciljne enote z sBERT: {sbert_avg_rank:.2f}")