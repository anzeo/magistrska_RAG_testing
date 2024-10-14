import yaml
import search_tfidf as search_tfidf
import search_sBERT as search_sbert
import matplotlib.pyplot as plt


def get_tfidf_performance():
    avg_rank = 0
    ranks = []

    enote, tfidf_matrix, vectorizer = search_tfidf.prepare_data()

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        top_indices, _ = search_tfidf.search(query, tfidf_matrix, vectorizer)

        target_unit_rank = 1
        for idx in top_indices:
            if enote[idx] == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_sbert_performance():
    avg_rank = 0
    ranks = []

    enote, preprocessed_enote_embeddings = search_sbert.prepare_data()
    
    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        top_indices, _ = search_sbert.search(query, preprocessed_enote_embeddings)

        target_unit_rank = 1
        for idx in top_indices:
            if enote[idx] == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


if __name__ == '__main__':
    with open('test_set.yaml', 'r') as file:
        test_data = yaml.safe_load(file)    

    tfidf_avg_rank, tfidf_ranks = get_tfidf_performance()
    sbert_avg_rank, sbert_ranks = get_sbert_performance()

    print(f"Povprečen rank ciljne enote s TF-IDF: {tfidf_avg_rank:.2f}")
    print(f"Povprečen rank ciljne enote z sBERT: {sbert_avg_rank:.2f}")

    plt.figure(figsize=(8, 6))  # Set the figure size

    # Use boxplot function to create the plot with labels
    plt.boxplot([tfidf_ranks, sbert_ranks], patch_artist=True, tick_labels=['TF-IDF', 'sBERT'])

    # Add title and labels
    plt.title('Primerjava povprečnega ranka ciljne enote')
    plt.ylabel('Rank')
    plt.xlabel('Metoda')

    # Display the plot
    plt.show()