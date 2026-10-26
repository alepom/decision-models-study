# Sintesi per l'autore

**Titolo:** *When Are Decision Models Worth It? An Empirical Comparison with Small Task-Specific Classifiers* — Alessio Pomaro.
Preprint, non sottoposto a revisione tra pari.

## La domanda
Stanno nascendo molti "decision model" sulla scia di Jev: servono davvero, o un modello piccolo addestrato sul proprio
compito, magari in locale, ottiene lo stesso risultato?

## I test, spiegati in modo semplice
Abbiamo messo a confronto un **"esperto esterno"** (il decision model: legge le istruzioni ma non ha mai visto i tuoi messaggi) e
un **"apprendista interno"** (un modello piccolo che impara dagli esempi già smistati), con sei domande:
1. **Quanti esempi servono all'apprendista?** 5 per categoria, 20 per categoria, oppure tutti (3 ripetizioni ciascuno).
2. **Quanto contano le istruzioni?** Solo il nome della categoria, oppure nome e breve descrizione, sugli stessi messaggi.
3. **Da dove viene la bravura dell'esperto?** Elephant contro lo stesso modello Qwen "di partenza", senza addestramento decisionale.
4. **Si fanno ingannare?** Messaggi trappola, negazioni, cambi di idea, refusi, richieste vaghe o fuori tema.
5. **Sanno dire "non sono sicuro"?** Quanto è affidabile la loro sicurezza per mandare i casi dubbi a una persona.
6. **Quanto costano e quanto sono veloci?**

Dati: Banking77 (inglese, pubblico), MASSIVE (italiano, pubblico), Ottavo Store (italiano, sintetico, creato per lo studio).

## Cosa abbiamo trovato
- **Pochi dati:** i decision model vincono nettamente anche usando solo i nomi delle categorie, cioè senza nessun esempio
  etichettato. Con 5 esempi per categoria il miglior modello piccolo (SetFit) resta indietro di 21–24 punti; con 20 esempi di 7–17.
- **Attenzione alle descrizioni:** per Banking77 e MASSIVE le descrizioni delle categorie sono state scritte leggendo esempi
  etichettati del training. Quindi il confronto "senza etichette" corretto è quello con i soli nomi; ora è il confronto principale
  del paper (segnalato da una revisione esterna e dichiarato come aggiunta successiva).
- **Tutti i dati:** il modello piccolo si avvicina molto. Con la ricetta del protocollo e rispetto a Mammoth con i soli nomi: su
  Banking77 0,910 contro 0,934 (2,4 punti), su MASSIVE 5,5 punti sotto, su Ottavo differenza non risolta (0,975 contro 0,975).
- **La ricetta conta:** su Banking77, con la ricetta dello studio esplorativo (4 seed), Qwen3-0.6B sale a 0,936: rispetto a
  Mammoth (0,925 con descrizioni, 0,934 con i soli nomi) nessuna delle due differenze è risolta. "Non risolta" non vuol dire "pari":
  il test non distingue i due sistemi, ma non dimostra che siano equivalenti. Le altre combinazioni di dataset e
  modello non sono state riprovate con questa ricetta, quindi i loro distacchi potrebbero essere sovrastimati.
- **Le descrizioni delle categorie** hanno effetti diversi per modello e dataset: aiutano OpenAI su Banking77 e tutti su Ottavo;
  altrove gli effetti sono piccoli e misti (per esempio +0,037 per Elephant su MASSIVE, non significativo dopo la correzione).
- **Elephant è molto più vicino al suo modello base** (Qwen3.5-9B, già 0,974 senza addestramento contro 0,987) che a un generalista
  più vecchio: il distacco residuo è piccolo sul test normale e circa 3 punti sui messaggi trappola. Non è una scomposizione causale.
- **Elephant e Mammoth** non cambiano risposta se si cambia l'ordine delle opzioni e hanno probabilità affidabili;
  **OpenAI** è sensibile all'ordine e le sue probabilità sono "a gradini", meno utili per decidere cosa mandare a una persona.
- **Con 5 esempi** i modelli Qwen addestrati con una testa di classificazione crollano con entrambe le ricette provate;
  **SetFit** è il migliore fra i metodi compatti valutati in quel regime.

## Cosa cambia rispetto al report iniziale (artifact)
| Affermazione iniziale | Ora |
|---|---|
| "Con 10.000 esempi Qwen3-0.6B (0,940) batte Mammoth (0,935)" | **Corretta.** Il punteggio si riproduce: con la ricetta originale e 4 seed fa 0,936 (0,931–0,940). Ma "batte" no: rispetto a Mammoth la differenza non è risolta. La ricetta del protocollo (0,910) era semplicemente peggiore su questo caso. |
| "Il crollo con 5 esempi è colpa della ricetta" | **Ritirata.** Con la ricetta scelta in cross-validation il risultato è identico (0,150 contro 0,150). È un limite del metodo con testa di classificazione nuova. |
| "Le descrizioni contano più del modello" | **Ridimensionata.** L'effetto varia con modello e dataset; inoltre sui dataset pubblici le descrizioni sono state scritte da esempi etichettati. |
| "Il vantaggio dei decision model è l'addestramento decisionale (6,2 milioni di esempi)" | **Ridimensionata.** Gran parte viene dal modello base più recente; l'addestramento decisionale aggiunge 1–3 punti. |
| "Elephant e Qwen3-8B differiscono solo per l'addestramento" | **Falsa.** Modello base diverso, attenzione modificata, obiettivo diverso. |
| "Una persona controlla solo i casi poco sicuri" | **Solo in parte.** Funziona su Banking77 (Mammoth accetta l'84% con errore ≤2%), non su MASSIVE-it (≤39%). |
| "La distillazione elimina il costo di etichettatura" | **Non verificata.** Resta un'idea per il futuro. |
| "Decision model gratis / modello locale costa centesimi" | **Ritirata.** Baseten era una promozione; l'elettricità non è stata misurata. |
| "~4.500 messaggi di test" | **Corretta:** 3.557 (campioni storici). |
| "Test avversario = robustezza" | **Rinominato** "stress lessicale": stesso compito e stessi generatori. |

## Limiti principali
Nessun messaggio reale e nessuna etichetta umana; dati Ottavo sintetici (scritti e validati da modelli Claude); possibile
contaminazione dei benchmark pubblici; endpoint in beta (ottobre 2026); test da ~1.000 messaggi.

## Abstract divulgativo (copiabile)
> I nuovi "decision model" promettono di classificare testi senza bisogno di esempi. Ho confrontato OpenAI Decisions, Elephant-1 e
> Mammoth-1 con modelli piccoli addestrati sul compito, su dati pubblici in inglese e italiano e su un compito sintetico di assistenza
> clienti. Con pochi esempi etichettati i decision model vincono nettamente; con migliaia di esempi i modelli piccoli si avvicinano
> molto ma non li superano. Gran parte della qualità dei decision model viene dal loro modello di base, e le descrizioni delle
> categorie aiutano alcuni sistemi ma non altri. È un preprint non sottoposto a revisione tra pari, con codice e predizioni pubblici.
