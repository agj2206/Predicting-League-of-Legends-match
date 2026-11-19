# Parte B → Model Evaluation

Questo documento accompagna `README_PART_B.md`. Nessun risultato del Test Set
viene calcolato nella Parte B; la futura Model Evaluation resta da svolgere.

## Cosa ricevi

Il run predefinito è `diamond_5x3_seed42`. In `reports/model_training/<run_id>/`
trovi `run_metadata.json`, `selection.json`, `best_params.json`, `model_comparison.json`,
`nested_cv_scores.json`, i tre `cv_results_*.json` finali e i 15 risultati in `inner_cv/`.
I CSV duplicano questi risultati e possono essere condivisi localmente.

`models/part_b/<run_id>/` contiene quattro Pipelines già fittate sulle 7.903 partite
di training: `dummy`, `decision_tree`, `knn`, `svm`. KNN e SVM includono lo scaler;
gli altri due non lo richiedono. Il singolo artefatto SVM contiene il kernel scelto
fra Linear e RBF tramite la Grid Search finale. Non occorre un secondo fit.

`data/processed/part_b/<run_id>/cv_manifest.json` conserva gli ID dei fold del solo
Training Set. Il manifest della Parte A rimane in `data/processed/split_manifest.json`.
Le due famiglie di manifest non contengono label o feature.

## Preparazione e caricamento

Usa lo stesso dataset Diamond, Python 3.11 e le versioni di
`requirements-part-b.txt`. Esempio di caricamento, da eseguire dalla repository:

```python
from src.lol_preprocessing import prepare_data
from src.lol_model_training import load_final_model, load_run_results

run_id = "diamond_5x3_seed42"
results = load_run_results(run_id=run_id)
selected = results["metadata"]["selected_model"]
model = load_final_model(selected, run_id=run_id)
prepared = prepare_data()  # riottiene lo split originale, senza rifittare nulla
```

Queste istruzioni non eseguono Model Evaluation. Quando inizierà quella fase,
usa `prepared.X_test` **non scalato** e con le colonne originali nell'ordine
registrato. Passalo direttamente a `model.predict`, senza `fit`, `fit_transform`
o nuovi scaler. Per ROC-AUC usa `decision_function` quando disponibile (SVM),
altrimenti `predict_proba` selezionando la colonna della classe `1`.

La target positiva è `blueWins=1` (Blue wins). Precision, Recall e F1 della CV
sono binary con `zero_division=0`. Per la valutazione finale puoi aggiungere
Confusion Matrix e risultati per entrambe le classi, mantenendo le cinque metriche
comuni. L'Accuracy è il criterio principale dichiarato prima dell'esperimento.

Controlla che `run_metadata.json` abbia `status="complete"` e
`held_out_test_used=false`, e che il checksum sorgente sia quello originale.
`load_final_model` verifica checksum del modello, schema delle feature, versioni
dei pacchetti e Python major/minor. Se mancano i binari, i JSON aggregati sono
ancora consultabili ma non bastano per fare predizioni: rigenera gli artefatti.

## Regole della valutazione finale

La famiglia scelta è congelata in `selection.json`, sulla base della Outer mean
Accuracy di Nested CV 5×3, prima dei tuning finali. I tre algoritmi appresi hanno
ricevuto lo stesso budget di fold e le griglie documentate; la Baseline non è tunata.
Il confronto finale sul test può includere tutti e quattro i modelli, ma non
deve essere usato per cambiare il vincitore, gli Hyperparameters, le soglie,
le feature o le regole di preprocessing.

`cv_results_*` descrive la Grid Search finale a 3 fold, usata per scegliere i
parametri. Il campo `split*_test_*` di sklearn indica un Validation Fold interno,
non il Test Set indipendente. Per presentare il confronto CV usa invece
`model_comparison.json` e `nested_cv_scores.json` (Outer Folds).
Le SD sono sample SD fra cinque fold, non Confidence Intervals.

Se i modelli sono vicini, evita affermazioni di superiorità certa. Il test fornirà
una stima finale per il campione riservato; non dimostrerà generalizzazione ad
altri rank, giocatori o patch. Non addestrare sui 9.879 record prima di valutare:
perderesti l'indipendenza del Test Set.

## GitHub e passaggio dei file ignorati

La `.gitignore` originale esclude `.joblib`, tutti i `.csv` e `data/`. Non è stata
modificata. I JSON aggregati, questo documento, `README_PART_B.md`, i test, il
codice e il notebook eseguito possono essere condivisi successivamente via GitHub.
Nessun CSV originale, riga del dataset o predizione individuale compare nei JSON
pubblicabili o nel notebook della Parte B.

Per rigenerare, segui i comandi in `README_PART_B.md`: test rapidi, training,
esecuzione del notebook. Se il run condiviso è già presente, scegli un nuovo
`--run-id` e usa lo stesso ID nel caricamento. I vecchi risultati non vengono
sovrascritti; non cambiare seed o griglia per ottenere un risultato più favorevole.

In alternativa, il membro che ha eseguito il training può passarti privatamente
le cartelle `models/part_b/<run_id>/`, `reports/model_training/<run_id>/` e
`data/processed/part_b/<run_id>/`, preservando i percorsi relativi. Non occorre
includere i binari in Git né modificare `.gitignore`. Carica soltanto artefatti
`joblib` provenienti dal membro del gruppo previsto. Il dataset originale deve
essere disponibile localmente anche al destinatario.
