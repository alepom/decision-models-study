# Adversarial set generator A

Model(s): claude-sonnet-5-5

## Message 1 to the agent

Stai scrivendo un set di TEST AVVERSARIO per un benchmark di classificazione di messaggi di assistenza clienti in italiano per "Ottavo Store" (e-commerce italiano inventato di elettronica/casa, 12 negozi fisici con click & collect, tessera fedeltà "Ottavo Club", pagamenti con carta/PayPal/bonifico/rate).

Leggi la tassonomia completa (24 classi con definizioni) in /home/alepom/progetti/lab-test-AP/classificatori/data/ottavo/labels.json. Le definizioni sono la regola: l'etichetta corretta è quella che un operatore esperto assegnerebbe applicando QUELLE definizioni all'intento PRINCIPALE del cliente (cosa chiede di fare/sapere).

Un primo set di test si è rivelato troppo facile (i classificatori fanno 99%). Ora servono messaggi DIFFICILI: ogni messaggio deve stare vicino al confine tra due classi simili, contenere segnali lessicali che "spingono" verso la classe sbagliata, e tuttavia avere UNA sola etichetta corretta secondo le definizioni. Tecniche utili: il cliente usa le parole tipiche dell'altra classe (es. dice "voglio fare il reso" ma il prodotto è arrivato rotto); racconta una lunga storia sull'altra classe e chiede la sua cosa solo alla fine; usa negazioni ("non è un problema di consegna, è che..."); dettagli temporali decisivi (giorni passati dalla consegna, data promessa superata o no); linguaggio vago e colloquiale; refusi.

Per ciascuna delle 5 coppie qui sotto scrivi 20 messaggi per ciascun lato (40 per coppia, 200 totali):
1. ordine_stato_spedizione vs ordine_ritardo_reclamo
2. recesso_ripensamento vs reso_garanzia_difetto
3. danno_alla_consegna vs reso_garanzia_difetto
4. danno_alla_consegna vs articolo_errato_mancante
5. ordine_annullamento vs ordine_modifica_indirizzo

Scrivi ogni messaggio a mano (niente template o generazione combinatoria). Varia lunghezza (da 5 a 100 parole), registro e struttura; non ripetere schemi. Evita i casi davvero ambigui: se tu stesso esiti, riscrivilo.

Salva in /home/alepom/progetti/lab-test-AP/classificatori/data/ottavo/adv_raw/gen_A.jsonl, una riga per messaggio: {"text": "...", "label": "<classe corretta>", "pair": <numero coppia>}. Scrivi a blocchi appendendo. Alla fine verifica che ci siano ~200 righe JSON valide e rispondi solo con il conteggio per label.

---

## Message 2 to the agent

<system-reminder>
Your final report is delivered through SubagentHandback: when your work is complete, call SubagentHandback({message: <your full report>}). The call ends your run, so make it your last step. Only a SubagentHandback call reaches your caller as your result; plain text you write at the end is not delivered. Do not also send the report with SendMessage: your caller would get it twice.
</system-reminder>