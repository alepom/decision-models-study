# Generate synthetic data group 0

Model(s): claude-sonnet-5-5

## Message 1 to the agent

Stai creando un dataset sintetico PRIVATO per un benchmark di classificazione di messaggi di assistenza clienti in italiano. Il negozio è inventato: "Ottavo Store", e-commerce italiano di elettronica, casa e piccoli elettrodomestici (vende anche in 12 negozi fisici con ritiro click & collect, ha una tessera fedeltà "Ottavo Club", accetta carte, PayPal, bonifico e pagamento a rate).

La tassonomia completa (24 classi, con definizioni) è in /home/alepom/progetti/lab-test-AP/classificatori/data/ottavo/labels.json — leggila tutta: serve a capire i confini tra le classi.

IL TUO COMPITO: scrivere 110 messaggi per ciascuna di queste 4 classi:
{CLASSES}

Scrivi i messaggi TU direttamente (non usare script che generano testo con template o combinazioni: ogni messaggio deve essere scritto a mano, unico). Salva un file JSONL per classe in /home/alepom/progetti/lab-test-AP/classificatori/data/ottavo/raw/<nome_classe>.jsonl, una riga per messaggio: {"text": "...", "label": "<nome_classe>"}. Puoi scrivere i file a blocchi (es. 4-5 scritture da ~25 righe per classe, appendendo) per non fare output troppo lunghi.

Requisiti di realismo e varietà (importanti, il benchmark deve essere difficile e realistico):
- Mix di canali: chat brevi (3-15 parole), email medie (20-60 parole), qualche email lunga e prolissa (60-120 parole) con contesto irrilevante.
- Registri diversi: formale, colloquiale, arrabbiato, anziano poco pratico, giovane con abbreviazioni ("nn", "xké", "cmq"), qualche refuso e punteggiatura scarsa, qualche messaggio tutto minuscolo o con maiuscole urlate. Qualche regionalismo leggero.
- Dettagli concreti variabili: numeri d'ordine (formato tipo OT-2026-xxxxx, ma non sempre), nomi di prodotti realistici e generici (friggitrice ad aria, cuffie bluetooth, robot aspirapolvere, smart TV 55", lavatrice, tablet...), date, importi, città.
- NON ripetere lo stesso schema di frase: varia l'apertura, la struttura e il lessico. Evita di usare nel testo le parole esatte del nome della classe in più del ~30% dei casi (le persone reali non parlano come le etichette).
- Circa il 20% dei messaggi deve essere "difficile": menziona elementi di altre classi vicine ma l'intento PRINCIPALE è inequivocabilmente quello della tua classe (es. per un reclamo di ritardo può citare il tracking; per un recesso può dire che il prodotto funziona ma non piace). Rispetta rigorosamente le definizioni del file labels.json: ogni messaggio deve avere UNA sola etichetta corretta che un operatore esperto assegnerebbe senza dubbi.
- Nessun messaggio ambiguo o duplicato.

Alla fine, rileggi i tuoi file, elimina o correggi messaggi ambigui, verifica con un comando che ogni file abbia almeno 105 righe JSON valide, e rispondi solo con il conteggio per classe e 2 esempi "difficili" che hai scritto.

---

## Message 2 to the agent

<system-reminder>
Your final report is delivered through SubagentHandback: when your work is complete, call SubagentHandback({message: <your full report>}). The call ends your run, so make it your last step. Only a SubagentHandback call reaches your caller as your result; plain text you write at the end is not delivered. Do not also send the report with SendMessage: your caller would get it twice.
</system-reminder>

---

## Message 3 to the agent

The coordinator sent a message while you were working:
Correzione: nel prompt il segnaposto {CLASSES} non è stato sostituito. Le tue 4 classi sono: ordine_stato_spedizione, ordine_ritardo_reclamo, ordine_modifica_indirizzo, ordine_annullamento. Scrivi SOLO queste 4 (altri agenti scrivono le altre 20 classi in parallelo nella stessa cartella raw/: non toccare file di altre classi).

Address this before completing your current task.