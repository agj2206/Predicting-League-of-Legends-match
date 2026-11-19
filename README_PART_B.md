# Parte B — Model Training (CDS3004)

**Research Question:** Can we predict the winning team in a League of Legends match
using only statistics available after the first 10 minutes?

Si usa esclusivamente il CSV Diamond originale e il Preprocessing della Parte A,
senza modificarne codice, feature, split o EDA. La target è `blueWins`: `1` = Blue
wins; `0` = Red wins. I 12 predictor sono differenze Blue − Red a 10 minuti.

## Riproduzione locale

Usare Python 3.11 e l'ambiente `.venv` esistente. Le versioni aggiuntive della
Parte B sono fissate in `requirements-part-b.txt`, che include il file originale
`requirements.txt`. Non è necessario eseguire nuovamente il notebook della Parte A.

Se occorre preparare un ambiente su un altro computer, installare le dipendenze
con `python -m pip install -r requirements-part-b.txt` nel proprio virtual environment.
Il training non scarica dati né installa pacchetti.

Il membro del gruppo deve procurarsi localmente lo stesso file
`high_diamond_ranked_10min.csv`, senza rinominarlo. Può collocarlo in `data/raw/`,
usare la cartella `Local_Datasets` accanto alla repository, oppure passare
`--data-path /percorso/high_diamond_ranked_10min.csv`.
`LOL_DATA_PATH` è supportata dall'interfaccia della Parte A.
Il checksum atteso è
`82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`.
Non cambiare il checksum o lo split per accettare un dataset diverso.

Dalla radice della repository:

```bash
# 1. Test rapidi della Parte B, su un sottoinsieme del Training Set Diamond.
.venv/bin/python -B -m unittest discover -s tests -p test_model_training.py -v

# 2. Training completo: 5 Outer Folds, 3 Inner Folds; solo training data.
.venv/bin/python -B scripts/train_models.py --n-jobs 2

# 3. Notebook con risultati, spiegazioni e grafici, senza rifare il training.
.venv/bin/python -B scripts/execute_training_notebook.py

# 4. Suite completa, inclusi i test originali della Parte A.
.venv/bin/python -B -m unittest discover -s tests -v
```

Il run predefinito si chiama `diamond_5x3_seed42`. I risultati esistenti non vengono
sovrascritti. Per una verifica indipendente usare, per esempio:

```bash
.venv/bin/python -B scripts/train_models.py --run-id diamond_reproduction --n-jobs 2
.venv/bin/python -B scripts/execute_training_notebook.py --run-id diamond_reproduction
```

`--n-jobs 1` permette un'esecuzione senza worker paralleli. Il loop esterno rimane
sempre sequenziale; anche i thread interni di BLAS/OpenMP sono limitati. `n_jobs`
non cambia i fold, le griglie o il criterio di selezione.
Un run interrotto ha stato `running` o `failed`: non deve essere usato come risultato
finale. Per ripartire scegliere un nuovo `--run-id`; non c'è resume automatico.

## Metodo e scelta del modello

`load_training_data()` richiama `prepare_data()` e restituisce soltanto `X_train`,
`y_train` e la provenienza. Il CSV contiene 9.879 partite, ma questa fase usa solo
le 7.903 partite del Training Set. Lo split 80/20 originale, le feature e
`random_state=42` restano fissi. La preparazione crea gli oggetti test come parte
del contratto esistente e può verificarne gli ID nel manifest; i valori di
`X_test` e `y_test` non sono passati a fit, scoring, tuning o selezione.

La Nested Stratified Cross-Validation usa 5 Outer Folds e 3 Inner Folds, con
`shuffle=True`, `random_state=42`. Per ciascun Outer Fold, `GridSearchCV` sceglie
gli Hyperparameters sui soli Inner Folds. Il miglior modello viene rifittato
sull'Outer Training Fold e valutato sull'Outer Validation Fold, escluso dal tuning.
Gli stessi fold sono usati per tutti gli algoritmi.

| Famiglia | Pipeline e Hyperparameters | Configurazioni |
|---|---|---:|
| Baseline | `DummyClassifier(strategy="most_frequent", random_state=42)` | Nessun tuning |
| Decision Tree | Nessuno scaling; `criterion=[gini, entropy]`, `max_depth=[3,5,8,12,None]`, `min_samples_leaf=[1,5,10,20]` | 40 |
| KNN | `StandardScaler` → KNN; `n_neighbors=[3,5,11,21,41,81]`, `weights=[uniform,distance]`, Minkowski `p=[1,2]` | 24 |
| SVM | `StandardScaler` → SVC; Linear con `C=[0.01,0.1,1,10]`; RBF con gli stessi `C` e `gamma=[scale,0.01,0.1,1]` | 20 |

La Baseline predice la classe più frequente del proprio Training Fold. Non usa
le feature per apprendere relazioni con la target. È un riferimento diagnostico,
non partecipa alla selezione fra Decision Tree, KNN e SVM. Se i modelli appresi non
la superano, questo viene segnalato esplicitamente.

KNN e SVM imparano media e deviazione standard dentro ogni fit della Pipeline.
Non si scala mai tutto `X_train` prima della CV. Decision Tree e Baseline non
includono uno scaler. L7 propone anche scaling in `[0,1]`: qui si usa invece
`StandardScaler`, come richiesto, che non impone quell'intervallo.

La metrica primaria è **Accuracy**, perché le classi del training sono quasi
bilanciate e non sono definiti costi asimmetrici degli errori. Si registrano anche
**Precision, Recall, F1 e ROC-AUC**. La classe positiva è Blue (`1`);
Precision/Recall/F1 sono binary, non macro. In assenza di predizioni positive,
Precision/F1 sono riportate a zero con `zero_division=0`, senza warning nascosti.
Per ROC-AUC si usano probabilità della classe `1` per Tree/KNN/Dummy e
`decision_function` per SVM. `probability=False`: nessuna calibrazione o soglia
aggiuntiva viene appresa. Le altre metriche usano `predict`.

Il confronto usa media e **sample standard deviation (ddof=1)** sui 5 Outer Folds.
La SD descrive variabilità e non è un Confidence Interval. Vince la famiglia con
la maggiore Outer mean Accuracy. Parità esatte: SD minore, poi ordine fisso
Decision Tree → KNN → SVM. Nelle singole Grid Search, parità esatte sono risolte
con l'ordine deterministico di `ParameterGrid`.

La scelta viene salvata in `selection.json` **prima** delle ricerche finali.
Poi ogni famiglia effettua un'ultima Grid Search a **3 fold sul Training Set intero**,
con `refit="accuracy"`; la Baseline viene semplicemente fittata sul training intero.
Si salvano quattro modelli finali. Gli Inner scores finali servono al tuning e non
sono usati per cambiare la famiglia già selezionata tramite Outer CV.
Sono previsti 1.536 fit: 1.260 nei tuning esterni + 20 refit esterni, quindi 252 nel
tuning finale + 4 fit finali. Non si addestra sul Test Set né sul dataset completo.

## File e artefatti

| Percorso | Contenuto |
|---|---|
| `src/lol_model_training.py` | API riutilizzabile, Pipelines, CV, tuning, scoring, salvataggio e caricamento |
| `scripts/train_models.py` | Entry point del training completo |
| `scripts/execute_training_notebook.py` | Esecuzione e salvataggio del solo notebook B con l'interprete corrente |
| `notebooks/02_model_training.ipynb` | Metodo, risultati CV, figure e limiti; non legge il Test Set |
| `tests/test_model_training.py` | Test rapidi su record Diamond del solo Training Set |
| `requirements-part-b.txt` | Dipendenze aggiuntive fissate, senza modificare la Parte A |
| `reports/model_training_handoff.md` | Istruzioni per la futura Model Evaluation |
| `reports/model_training/<run_id>/` | Risultati e metadati del run |
| `models/part_b/<run_id>/` | `dummy.joblib`, `decision_tree.joblib`, `knn.joblib`, `svm.joblib` |
| `data/processed/part_b/<run_id>/cv_manifest.json` | ID dei record training assegnati a ogni fold; nessuna label o feature |

Ogni run salva `nested_cv_scores`, `model_comparison` e `cv_results_*` in **CSV e JSON**.
`inner_cv/` contiene tutte le 15 ricerche degli Outer Folds, sempre in entrambi i
formati. I risultati completi includono parametri, punteggi per fold, medie, SD,
rank per metrica e tempi. La colonna sklearn `split*_test_*` indica la validation
interna alla CV: **non è il Test Set held-out della Parte A**.

`best_params.json` contiene gli Hyperparameters finali; `selection.json` conserva
la scelta della famiglia; `run_metadata.json` registra stato, seed, griglie, schema,
versioni, fingerprint del training, checksum del sorgente, dei modelli e del manifest,
durata, warning e assenza di utilizzo del test. Tempi e timestamp non sono risultati
deterministici: la riproducibilità riguarda split, parametri, predizioni e metriche
nello stesso ambiente. I file binari non devono necessariamente essere identici
byte per byte fra piattaforme.

## Condivisione successiva tramite GitHub

Nessun comando Git è necessario per eseguire la Parte B. Questa implementazione
non modifica branch, index, commit o remote e lascia invariata la `.gitignore`.

La `.gitignore` esistente esclude **tutti i CSV**, i `.joblib` e `data/`. Perciò
i CSV, i modelli finali e il CV manifest locale non compariranno automaticamente
su GitHub. I JSON sotto `reports/model_training/`, il notebook eseguito, il codice,
i test e la documentazione sono invece condivisibili con le regole attuali.
I JSON contengono soltanto risultati aggregati e configurazioni, senza righe del
dataset o predizioni individuali; sono la copia completa dei risultati delle
ricerche, leggibile anche senza CSV. Gli ID dei fold restano nel manifest ignorato.

Per il membro della Model Evaluation sono possibili due modalità:

1. **Rigenerazione:** stesso CSV originale e Python 3.11, installazione delle
   dipendenze fissate, test rapidi, poi `scripts/train_models.py`. La procedura
   ricostruisce lo stesso split della Parte A e tutti gli artefatti della Parte B.
   Se sono già presenti risultati condivisi per quel run, usare un nuovo `--run-id`.
2. **Passaggio degli artefatti locali:** condividere privatamente, mantenendo i
   percorsi relativi, le tre cartelle `models/part_b/<run_id>/`,
   `reports/model_training/<run_id>/` e `data/processed/part_b/<run_id>/`.
   I CSV possono essere inclusi in questo passaggio. Per la futura valutazione
   occorre comunque il CSV originale, disponibile localmente al destinatario.

Non occorre forzare l'inclusione di modelli o dataset in Git. Prima di caricare
modelli ricevuti, verificare che provengano dal membro del gruppo previsto:
`joblib` serializza oggetti Python. `load_final_model()` controlla checksum, schema,
versioni dei pacchetti e Python major/minor prima del caricamento.

## Collegamento al corso e limiti

L4–L5 spiegano Classification, Decision Tree, Gini ed Entropy; L6, slide 25–26,
il Pre-pruning; L6, slide 44–51, Cross-Validation e Nested CV; L7, slide 9–24,
KNN, scaling, SVM, C e kernel; L7, slide 27–34, soglie, ROC e confronto equo.

Le performance CV sono stime sul campione Diamond con questa rappresentazione
di 12 feature. Scegliere la famiglia fra più candidati introduce ulteriore Selection
Bias: l'Outer score del vincitore non sostituisce una valutazione finale indipendente.
Cinque fold non dimostrano significatività statistica fra modelli vicini.
La griglia è finita; un optimum al bordo non prova un optimum globale e non
giustifica cambiamenti dopo aver visto il test. Scaling non elimina correlazioni
o valori estremi. Restano i limiti della Parte A: ward count sospetti, assenza di
player ID e patch/time metadata, omissione del ritmo assoluto. Nessun filtro,
Feature Selection, resampling o nuova EDA è stato aggiunto per aggirare questi limiti.

La futura Model Evaluation applicherà i modelli congelati al Test Set una volta.
Potrà confrontarli descrittivamente, ma non usare quei risultati per riaprire il
tuning o cambiare la famiglia già selezionata.
