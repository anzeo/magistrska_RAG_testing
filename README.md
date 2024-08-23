# Magistrska naloga

## Začetek
Za pravilno delovanje skript je priporočena uporaba `Python 3.11.9`.

Da se namestijo vse potrebne knjižnice za delovanje, je potrebno na začetku pognati sledeči ukaz:
```
pip install -r requirements.txt
```

## Skripte

- `parse_document.py` - skripta iz spleta pridobi vsebino akta o umetni inteligneci, nato pa dokument razdeli na krajše enote, ki jih v obliki posameznih točk in členov zapiše v .yaml formatu. Rezultat je shranjen v datoteki `ai_act.yaml`.

- `search_tfidf.py` in `search_sBERT.py` - prva skripta vsebuje metode za iskanje po dokumentu z uporabo metode TF-IDF, medtem ko druga skripta vključuje metode za iskanje z uporabo jezikovnega modela sBERT. 
Za testiranje posameznega iskalnega pristopa lahko uporabimo metodo `get_relevant_results()`, ki kot argument sprejme vprašanje. Kot rezultat se v konzoli izpišejo enote iz dokumenta, ki so najrelevantnejše glede na zastavljeno vprašanje. Po potrebi lahko dodamo še numerični parameter, s katerim določimo število prikazanih relevantnih enot. Če ta parameter izpustimo, se prikažejo vse enote.

- `compare_search_methods.py` - skripta omogoča primerjavo učinkovitosti iskanja relevantnih enot z metodama TF-IDF in sBERT. Na podlagi pripravljenih testnih primerov iz datoteke `test_set.yaml` (ki vsebuje nabor vprašanj s pripadajočimi ciljnimi enotami, kjer so odgovori), skripta izračuna in izpiše povprečni rank ciljne enote pri rangiranju z uporabo TF-IDF ali sBERT.\
**Zaenkrat kaže, da metoda TF-IDF na pripravljenem naboru testnih podatkov deluje bolje od sBERT, če primerjamo povprečni rank ciljne enote.**