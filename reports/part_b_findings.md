# Parte B: risultati misurati

Run: `diamond_5x3_seed42`. Dataset: **Diamond only**. Parte A riutilizzata senza modifiche.

**Research Question:** Can we predict the winning team in a League of Legends match using only statistics available after the first 10 minutes?

## Metodo

7.903 partite del Training Set, 12 feature non scalate in ingresso. Nested Stratified Cross-Validation con 5 Outer Folds e 3 Inner Folds, shuffle=True e random_state=42. Il Test Set non è stato usato per training, scoring, tuning o selezione.

Accuracy come criterio primario; classe positiva Blue (`1`). Media ± sample SD (ddof=1) sui 5 Outer Folds. I valori seguenti non sono Test Set metrics.

| Modello | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| dummy | 0.5009 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| decision_tree | 0.7158 ± 0.0206 | 0.7216 ± 0.0315 | 0.7051 ± 0.0441 | 0.7121 ± 0.0212 | 0.7940 ± 0.0132 |
| knn | 0.7162 ± 0.0145 | 0.7145 ± 0.0188 | 0.7193 ± 0.0218 | 0.7167 ± 0.0141 | 0.7986 ± 0.0130 |
| svm | 0.7295 ± 0.0167 | 0.7284 ± 0.0160 | 0.7302 ± 0.0215 | 0.7292 ± 0.0175 | 0.8089 ± 0.0136 |

## Scelta e Hyperparameters finali

**Famiglia selezionata: SVM**, sulla base della Outer mean Accuracy, congelata prima del tuning finale. Miglioramento sulla Baseline: **22.85 punti percentuali**.

| Modello | Hyperparameters finali |
|---|---|
| dummy | `{"classifier__strategy": "most_frequent"}` |
| decision_tree | `{"classifier__criterion": "entropy", "classifier__max_depth": 5, "classifier__min_samples_leaf": 20}` |
| knn | `{"classifier__n_neighbors": 81, "classifier__p": 1, "classifier__weights": "distance"}` |
| svm | `{"classifier__C": 10, "classifier__kernel": "linear"}` |

I parametri finali sono scelti con 3-fold CV sul Training Set intero. Tutti e quattro i modelli salvati sono già fittati su tutte le 7.903 partite di training. KNN e SVM includono StandardScaler nella Pipeline. La SVM finale usa Linear; RBF è stato comunque testato nella stessa ricerca.

## Spiegazione per la presentazione

La Baseline risponde sempre Red, la classe leggermente più frequente, e ottiene circa il 50%. I modelli appresi migliorano fino a circa il 73% con SVM: in media, nei Validation Folds, circa 73 vincitori corretti su 100 partite. Le informazioni dei primi dieci minuti mostrano quindi capacità predittiva nel campione Diamond esaminato; la conferma sul Test Set indipendente resta alla Model Evaluation.

La Baseline ha Precision, Recall e F1 pari a zero per Blue perché non predice mai Blue; non è un errore di calcolo. La sua ROC-AUC è 0.5. La ROC-AUC della SVM è circa 0.809: misura ordinamento, non una probabilità di vittoria del singolo match.

Decision Tree e KNN hanno Accuracy molto vicine. SVM è prima in tutti e cinque gli Outer Folds, ma ciò non costituisce una prova di significatività statistica o una garanzia su partite future.

## Limiti e problemi espliciti

- KNN sceglie k=81 e SVM sceglie C=10, entrambi al limite superiore della griglia. Non è dimostrato un optimum globale. La griglia resta quella predefinita e non viene ampliata guardando il test.
- Per SVM, la mean Accuracy del tuning finale con C=10 è 0.729596; con C=1 è 0.729216: la differenza è molto piccola. La regola di scelta rimane quella dichiarata, senza una nuova decisione a posteriori.
- Gli Outer scores hanno partecipato alla selezione della famiglia e non sostituiscono il test indipendente. Le SD non sono Confidence Intervals.
- Restano i limiti della Parte A: ward count sospetti, feature correlate, assenza di player ID e patch/time metadata, omissione del ritmo assoluto. Nessuna nuova pulizia, EDA o Feature Selection.

## Esecuzione e verifiche

Training completo: **1.536 fit**, circa **119.5 secondi** su questo computer; tempi dipendenti dalla macchina. Nessun runtime warning registrato.

Prima del training sono passati gli 8 test rapidi della Parte B. Successivamente sono passati tutti i 21 test della suite (8 Parte B + 13 Parte A). I test ridotti verificano due esecuzioni identiche, scaling per fold e fit limitati ai Training Folds. I modelli completi sono stati ricaricati e controllati, senza predizioni sul Test Set. I controlli sugli ID verificano esclusivamente la struttura degli split.

L'audit degli artefatti ha superato 58 controlli; dettagli in `model_training/diamond_5x3_seed42/artifact_verification.json`.

## Riproduzione e handoff

Comandi, file ignorati da Git, rigenerazione e passaggio privato delle cartelle sono documentati in [README_PART_B.md](../README_PART_B.md) e [model_training_handoff.md](model_training_handoff.md). I JSON aggregati conservano tutti i risultati CV anche quando CSV e modelli sono esclusi dalla .gitignore. L’elenco completo dei pacchetti installati è in `environment_versions.txt` nella cartella del run.

Il notebook `notebooks/02_model_training.ipynb` legge questi risultati, mostra tabelle e grafici e non rilancia il training. La Model Evaluation dovrà applicare le Pipelines a X_test non scalato, senza rifittarle o cambiare la selezione.

Il notebook è stato eseguito e salvato con tutte le 9 celle di codice completate,
un grafico incorporato e nessun error output. Il confronto SHA-256 iniziale/finale
ha confermato che tutti i 18 file preesistenti controllati sono rimasti identici,
inclusi codice, test, notebook, manifest e risultati della Parte A; dettagli in
`model_training/diamond_5x3_seed42/part_a_integrity.json`.

Problemi tecnici risolti: la sandbox inizialmente impediva le porte di comunicazione
locali di Jupyter; l'esecuzione del kernel locale è stata autorizzata separatamente.
La formattazione pandas Styler richiedeva Jinja2, non installata: è stata sostituita
con normale formattazione DataFrame senza installare pacchetti. La prima esecuzione
bloccata è stata chiusa. Nessuno di questi problemi ha modificato training o risultati.
