import os
import yaml
import matplotlib.pyplot as plt

TEST_DATA_DIR = 'data/test'
PLOTS_DIR = 'plots'

def get_tfidf_performance():
    import search_tfidf as search_tfidf

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
    import search_sBERT as search_sbert

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
    for test_data_filename in os.listdir(TEST_DATA_DIR):
        with open(os.path.join(TEST_DATA_DIR, test_data_filename), 'r') as file:
            test_data = yaml.safe_load(file)    

        print(f"Rezultati za podatke iz datoteke: {test_data_filename}")

        tfidf_avg_rank, tfidf_ranks = get_tfidf_performance()
        sbert_avg_rank, sbert_ranks = get_sbert_performance()

        print(f"Povprečen rank ciljne enote s TF-IDF: {tfidf_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote z sBERT: {sbert_avg_rank:.2f}")

        plt.figure(figsize=(8, 6))  # Set the figure size

        plt.boxplot([tfidf_ranks, sbert_ranks], patch_artist=True, tick_labels=['TF-IDF', 'sBERT'])

        plt.title(f'{test_data_filename}\nPrimerjava povprečnega ranga ciljne enote')
        plt.ylabel('Rank')
        plt.xlabel('Metoda')

        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_filename = f"{os.path.splitext(test_data_filename)[0]}_results_plot.svg"
        plt.savefig(os.path.join(PLOTS_DIR, plot_filename), format='svg')

        plt.draw()

        print('')
    plt.show()