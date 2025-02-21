import os

import numpy as np
import yaml
import matplotlib.pyplot as plt

TEST_DATA_DIR = 'data/test'
PLOTS_DIR = 'plots'

SEARCH_UNITS = ["cleni", "tocke"]


def get_tfidf_performance():
    import search_tfidf as search_tfidf

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_tfidf.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_sbert_performance():
    import search_sBERT as search_sbert

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_sbert.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_Llama_performance():
    import search_Llama as search_Llama

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_Llama.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_openai_performance():
    import search_openai as search_openai

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_openai.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_sloberta_performance():
    import search_sloberta as search_sloberta

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_sloberta.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def get_XLMRoberta_performance():
    import search_XLMRoberta as search_XLMRoberta

    avg_rank = 0
    ranks = []

    for test in test_data:
        query = test['vprasanje']
        target_unit = test['odgovor_enota']
        results = search_XLMRoberta.search(query, unit_conditions=SEARCH_UNITS)

        target_unit_rank = 1
        for element_id, score in results:
            if element_id == target_unit:
                break
            target_unit_rank += 1

        avg_rank += target_unit_rank
        ranks.append(target_unit_rank)

    return avg_rank / len(test_data), ranks


def draw_box_plot(all_ranks, filename):
    """
    Draws and saves a box plot to visualize the performance of different approaches based on their ranking scores.

    Args:
        all_ranks (dict): A dictionary with:

            - "ranks": A list of rank lists (one per approach).
            - "labels": A list of corresponding approach names.

        filename (str): The base filename for saving the plot.
    """

    plt.figure(figsize=(8, 6))  # Set the figure size

    plt.boxplot(all_ranks["ranks"], patch_artist=True, tick_labels=all_ranks["labels"])

    plt.title(f'{filename}\nPrimerjava povprečnega ranga ciljne enote')
    plt.ylabel('Rank')
    plt.xlabel('Metoda')

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = f"{os.path.splitext(filename)[0]}_box_plot.svg"
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename), format='svg')


def draw_cdf_plot(all_ranks, filename):
    """
    Draws and saves a CDF (Cumulative Distribution Function) plot of rank distributions.

    Args:
        all_ranks (dict): A dictionary containing:

            - "ranks": A list of rank lists (one per approach).
            - "labels": A list of corresponding approach names.

        filename (str): The base filename for saving the plot.
    """
    plt.figure(figsize=(8, 6))

    for ranks, label in zip(all_ranks["ranks"], all_ranks["labels"]):
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        plt.plot(sorted_ranks, cdf, label=label)

    plt.xlabel("Rank")
    plt.ylabel("Kumulativna verjetnost")
    plt.title(f"{filename}\nCDF ciljne enote")
    plt.legend()
    plt.grid()

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = f"{os.path.splitext(filename)[0]}_cdf_plot.svg"
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename), format='svg')


if __name__ == '__main__':
    plt.ion()
    for test_data_filename in os.listdir(TEST_DATA_DIR):
        with open(os.path.join(TEST_DATA_DIR, test_data_filename), 'r') as file:
            test_data = yaml.safe_load(file)

        tfidf_avg_rank, tfidf_ranks = get_tfidf_performance()
        sbert_avg_rank, sbert_ranks = get_sbert_performance()
        Llama_avg_rank, Llama_ranks = get_Llama_performance()
        openai_avg_rank, openai_ranks = get_openai_performance()
        sloberta_avg_rank, sloberta_ranks = get_sloberta_performance()
        XLMRoberta_avg_rank, XLMRoberta_ranks = get_XLMRoberta_performance()

        print(f"Rezultati za podatke iz datoteke: {test_data_filename}")

        print(f"Povprečen rank ciljne enote s TF-IDF: {tfidf_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote z sBERT: {sbert_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote z Llama: {Llama_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote z OpenAI: {openai_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote s sloberta: {sloberta_avg_rank:.2f}")
        print(f"Povprečen rank ciljne enote z XLM-Roberta: {XLMRoberta_avg_rank:.2f}")

        draw_box_plot({"ranks": [tfidf_ranks, sbert_ranks, Llama_ranks, openai_ranks, sloberta_ranks, XLMRoberta_ranks],
                       "labels": ['TF-IDF', 'sBERT', 'Llama', 'OpenAI', 'sloberta', 'XLM-Roberta']}, test_data_filename)

        draw_cdf_plot({"ranks": [tfidf_ranks, sbert_ranks, Llama_ranks, openai_ranks, sloberta_ranks, XLMRoberta_ranks],
                       "labels": ['TF-IDF', 'sBERT', 'Llama', 'OpenAI', 'sloberta', 'XLM-Roberta']}, test_data_filename)

        print('')
    plt.show(block=True)
