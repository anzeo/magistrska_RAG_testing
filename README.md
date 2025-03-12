# Magistrska naloga

## Navodila za uporabo

Za pravilno delovanje skript je priporočena uporaba `Python 3.11.9`.

Pred začetkom je potrebno namestiti vse zahtevane knjižnice z naslednjim ukazom:

```
pip install -r requirements.txt
```

Če želimo uporabljati skripto, ki temelji na **plačljivem OpenAI modelu**, je potrebno v root direktoriju ustvariti datoteko `.env` in vanjo dodati svoj OpenAI API ključ:

```
OpenAI_API_KEY={your_api_key}
```
Če api ključa nimamo, lahko izpustimo primerjavo OpenAI modela, tako da v skripti `compare_search_methods.py` zakomentiramo naslednjo vrstico kode:
```
openai_avg_rank, openai_ranks = get_openai_performance()
```

Po opravljenih zgornjih korakih lahko zaženemo skripto `compare_search_methods.py`, ki omogoča primerjavo različnih metod iskanja relevantnih enot na podlagi testnih podatkov.


## Skripte

### Razčlenjevanje akta

Skripta `parse_document.py` iz spleta pridobi vsebino akta o umetni inteligenci, nato pa dokument razdeli na krajše enote. Te enote so organizirane v obliki posameznih točk in členov ter shranjene v `.yaml` formatu v datoteki `ai_act.yaml`.

### Primerjava metod za iskanje in vektorsko predstavitev besedila

Ta zbirka Python skript implementira različne metode za ekstrakcijo vektorskih predstavitev besedila in iskanje po dokumentih. Vsaka skripta uporablja drugačen pristop za vektorizacijo in ocenjevanje podobnosti med poizvedbo in dokumenti.

#### Opis skript

- `search_tfidf.py` - Implementacija metode TF-IDF za izračun vektorskih predstavitev besedila in iskanje relevantnih dokumentov.
- `search_sBERT.py` - Uporaba modela `paraphrase-multilingual-mpnet-base-v2`, iz **sBERT** (Sentence-BERT), optimiziranega za večjezično razumevanje besedil.
- `search_Llama.py` - Implementacija iskanja s pomočjo modela **Llama**.
- `search_XLMRoberta.py` - Uporaba modela `xlm-r-100langs-bert-base-nli-stsb-mean-tokens`, iz **sBERT** (Sentence-BERT), optimiziranega za večjezično razumevanje besedil.
- `search_sloberta.py` - Uporaba **Sloberta** modela, prilagojenega za slovenski jezik.
- `search_openai.py` - Uporaba plačljivega **OpenAI** modela `text-embedding-3-small`.

Vse skripte vsebujejo pristope za izračun vektorskih predstavitev besedila (**vector embeddings**) in omogočajo ocenjevanje podobnosti med poizvedbami in dokumenti.

#### Uporaba

Vsaka skripta vsebuje metodo `get_relevant_results()`, ki omogoča testiranje posameznega iskalnega pristopa. Funkcija sprejme vprašanje kot vhodni parameter in vrne najrelevantnejše enote iz dokumenta.

**Primer klica funkcije:**

```python
results = get_relevant_results("Kako deluje metoda TF-IDF?", top_n=5)
```

Kjer je `top_n` število prikazanih relevantnih rezultatov. Če ga ne podamo, se prikažejo vse relevantne enote.

- `compare_search_methods.py` - Skripta omogoča primerjavo učinkovitosti iskanja relevantnih enot z vsemi implementiranimi metodami. Na podlagi pripravljenih testnih primerov iz datotek `test_set.yaml` in `test_set_teacher.yaml` (ki vsebujeta nabor vprašanj s pripadajočimi ciljnimi enotami) skripta izračuna in izpiše povprečni rang ciljne enote pri rangiranju z uporabo posameznih metod.