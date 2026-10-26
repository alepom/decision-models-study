# Stato del lavoro e passaggio di consegne (aggiornato 25/10/2026 notte)

Documento per riprendere il lavoro in una nuova sessione senza la cronologia della chat. Fonte di verità: git
(`git log --oneline`), `protocol.md`, `reports/`, `results/`.

## Mandato
Applicare `Relazione_operativa_Claude_Code_Decision_Models.md` (E00–E12): paper inglese LaTeX su Zenodo (preprint),
repository GitHub, due record Zenodo (paper e software), guide in italiano per l'autore (Alessio Pomaro).
Nessuna azione pubblica senza istruzione esplicita dell'autore.

## Decisioni prese con l'autore
- Budget OpenAI: tetto 10 $ ("soluzione bilanciata"); speso 8,10 $; **nessuna altra spesa OpenAI prevista**.
- Niente dati reali né annotatori: E06 bloccato, preparati solo i template (`data/annotation_templates/`).
- Disco: ok fino a ~15 GB, poi ripulire (ambiente `.venv`, modelli HF scaricati per il test).
- Calcolo GPU spostato interamente su **Colab T4** (la RTX 4050 laptop è limitata a 25 W e bloccava il PC);
  il portatile ha eseguito solo TF-IDF (CPU). Colab lo lancia l'autore dal suo **account Google personale**.
- Tectonic installato in `~/.local/bin` con autorizzazione dell'autore.
- L'autore preferisce restare nella stessa conversazione; il connettore Drive si è perso al riavvio della sessione:
  aggiornamenti tramite screenshot e zip finale.

## Stato degli esperimenti (25/10/2026)
Tutti gli esperimenti previsti sono conclusi: 478 run salvati, 450 usati nell'analisi (`results/runs_index.csv`, colonna `used`).
Dettaglio per attività in `reports/experiment_status.csv`.

| Attività | Stato |
|---|---|
| E00 audit, E01 protocollo | fatti; protocollo congelato (tag `protocol-frozen-v1`), addendum Ottavo-R |
| E02/E05 supervisionati | fatti su Colab T4 (TF-IDF in locale), 3 seed × 3 budget × 3 dataset; Qwen3-1.7B e Qwen3-8B solo su Ottavo |
| E03 descrizioni, ordine, esempi nel prompt, ripetizioni | fatti |
| E04 progenitore | fatto (Qwen3.5-9B e Qwen3-8B zero-shot su Ottavo, stress, Ottavo-R) |
| E07 stress lessicale e Ottavo-R | fatti |
| SENS ricetta esplorativa su Banking77 | fatta (4 seed di training, famiglia S1 dichiarata nel commit ee3feacc prima dei run) |
| E06 dati reali, E11 distillazione | non eseguiti (niente dati reali), dichiarati nei limiti |
| E12 paper e pubblicazione | paper completo (18 pagine); materiali in `publication/`; pacchetti da generare |

## Risultati principali (verificati, `reports/contrasts.csv`)
- Le descrizioni D di Banking77 e MASSIVE-it sono state scritte leggendo il training etichettato: il confronto a zero etichette usa
  N (famiglia P2N, aggiunta il 26/10 dopo una revisione esterna; P2 con D resta riportata).
- k=5, con N: Mammoth supera SetFit di +0,217 / +0,242 / +0,209 (Banking77 / MASSIVE-it / Ottavo); con D +0,207 / +0,249 / +0,224.
- Tutti i dati, ricette del protocollo, con N: Banking77 +0,024 [0,011; 0,039] su Qwen3-0.6B; MASSIVE-it +0,055 su mmBERT;
  Ottavo −0,001 [−0,012; 0,010] su Qwen3-8B (non risolta).
- **Sensibilità (S1):** con la ricetta esplorativa Qwen3-0.6B su Banking77 fa 0,936 (0,931–0,940), +0,026 [0,018; 0,035] sulla
  ricetta del protocollo, differenza con Mammoth **non risolta** (D −0,011 [−0,025; 0,002], N −0,002); non è equivalenza. Lo 0,940 storico si riproduce; "batte" no.
  La ricetta del protocollo per il regime "tutti i dati" era peggiore; non riprovato su altri dataset/modelli.
- Il crollo a 5 esempi dei decoder non dipende dalla ricetta (P5: +0,000 e +0,004).
- Progenitore: Elephant 0,987 vs Qwen3.5-9B zero-shot 0,974 su Ottavo (+0,013), +0,030 sullo stress.
- Descrizioni: +0,071 per OpenAI su Banking77, ≈ −0,01 per Elephant/Mammoth sui pubblici.
- Routing al 2% di errore su Banking77: Mammoth 84%, Elephant 79%, Qwen3-0.6B 80% (89% con la ricetta esplorativa), OpenAI 0%.
- Spesa OpenAI totale: 8,10 USD.

## Da fare
1. `scripts/publication.py export`, `package`, `check`; verifica del PDF nei pacchetti.
2. Decisioni dell'autore in `publication/AZIONI_RICHIESTE_ALL_AUTORE.md`; poi DOI, `publication.py set`/`render` e nuovi pacchetti.
3. Pulizia disco: `.venv`, modelli HF scaricati per il test (non e5-base, preesistente), `checkpoints/` (3,2 GB, vecchia prova SetFit
   locale), file pip di oggi.

## Aggiornamento 24/10/2026 sera (dopo il passaggio a Colab)
- Colab v3 (`notebooks/colab_runner.ipynb`, commit 734d4ae): 135 job; alle 21 circa 36 completati.
- Errore risolto: Colab ha `torchao` 0.10 preinstallato e `peft` recente rifiuta di partire (LoRA). Il notebook ora
  disinstalla torchao e blocca le versioni: transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2,
  scikit-learn 1.9.1, setfit 1.2.0, sentence-transformers 6.2.0, datasets 5.1.0. **I job Colab precedenti (zero-shot,
  SetFit, mmBERT) hanno usato le versioni più recenti non bloccate: dichiararlo nel paper.**
- La coda ora salta i job finali se manca la selezione della ricetta, riprova in fp32 in caso di NaN e stampa la coda
  del log quando un job fallisce.
- Ricette selezionate in CV (seed 17, Ottavo): mmBERT lr 5e-5, 20 epoche (k=5) e 12 (k=20) = **bordo della griglia**
  (da dichiarare tra i limiti); Qwen3-0.6B lr 2e-4, 10 epoche (k=5) e 12 (k=20). La ricetta originale usava 34 epoche a k=5.
- SetFit: max_seq_length 256 (cambiato prima dei run definitivi; i run a 512 sono superati e la pipeline usa l'ultimo run).
- Tempi T4 misurati: SetFit k5 ~4 min, k20 ~11 min, full ~19 min; mmBERT 1.5-3 min; Qwen0.6B CV 6.5/13.4 min, run ~4-6 min;
  zero-shot 8B/9B 8-17 min. Stima totale coda ~28 h di T4.
- Portatile: solo TF-IDF (finito, 27/27). La RTX 4050 (limite 25 W) era troppo lenta e bloccava il PC.
- Il connettore Google Drive non si è ricaricato dopo il riavvio della sessione: aggiornamenti via screenshot; risultati
  via `dm_results_v2.zip` nei Download dell'utente.

## Decisioni editoriali (24/10/2026, tarda sera)
- Titolo: **"When Are Decision Models Worth It? An Empirical Comparison with Small Task-Specific Classifiers"**
  (applicato a `paper/main.tex` e `scripts/publication.py`). Autore unico: **Alessio Pomaro** (confermato).
- L'introduzione parte dalla domanda dell'autore (decision model sulla scia di Jev: servono, o bastano modelli piccoli
  locali/fine-tunati?). Jev citato (VentureBeat) e dichiarato non testato (accesso su lista d'attesa).
- Sede: Zenodo come sede principale (paper + software), GitHub per il codice. Aggiunti alla guida come passi facoltativi:
  dataset Ottavo su **Hugging Face Hub** (scheda in `publication/huggingface/README.md`), **articolo divulgativo** in
  italiano (`publication/articolo_divulgativo_IT.md`, da scrivere a risultati finiti), **arXiv** solo come secondo passo
  se l'autore ottiene un endorsement, **versione 2** con messaggi reali come base per una sede con revisione.
- Spiegazione semplice dei test (da usare nella sintesi italiana e nell'articolo): "esperto esterno" (decision model,
  legge le istruzioni, non ha visto i messaggi) contro "apprendista interno" (modello piccolo che impara dagli esempi);
  6 domande: quanti esempi servono; quanto contano le istruzioni; da dove viene la bravura di Elephant (vs Qwen di partenza);
  si fanno ingannare; sanno dire "non sono sicuro"; quanto costano e quanto sono veloci.
