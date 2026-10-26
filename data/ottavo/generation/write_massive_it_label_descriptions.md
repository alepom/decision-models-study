# Write MASSIVE-it label descriptions

Model(s): claude-opus-5-5

## Message 1 to the agent

Task: write category descriptions ("condition D") for the 60 intents of the MASSIVE dataset, Italian locale (it-IT), for a research study on intent classification. They will be given to zero-shot classifiers together with the (English) label names, and the messages are in Italian, so write the descriptions in ITALIAN.

Use ONLY these two files (do not open any other file or directory; in particular never look for test data):
- /tmp/claude-1000/-home-alepom-progetti-lab-test-AP-classificatori/d1fea9b4-435b-421a-9137-8a40df42cf07/scratchpad/desc/massive_it/labels.json  (label -> humanized name)
- /tmp/claude-1000/-home-alepom-progetti-lab-test-AP-classificatori/d1fea9b4-435b-421a-9137-8a40df42cf07/scratchpad/desc/massive_it/train_full.jsonl  (training utterances: {"text","label"})
You may also use your general knowledge of the public MASSIVE dataset documentation (Amazon, intents in scenario_intent form), but do not search the web.

Method: for each label, read a sample of its training utterances (e.g. 20-40, use shell tools like grep/shuf), and especially compare easily confused labels (e.g. iot_hue_lightchange vs iot_hue_lightup vs iot_hue_lightdim, audio_volume_* variants, calendar_query vs calendar_set, general_quirky vs qa_factoid vs general_joke, music_query vs play_music, recommendation_* variants, email_query vs email_querycontact, lists_* variants, takeaway_order vs takeaway_query, transport_* variants).

Write for each label ONE description in Italian, 1-3 short sentences (max ~45 words): what the user asks the voice assistant to do or know (inclusion rule) and, where relevant, the boundary with the closest confusable labels ("non ...: usare X"). Be factual about what the training data shows (MASSIVE has some noisy/odd labels such as general_quirky: describe them honestly). Do not copy utterances verbatim.

Output: write a JSON object {label: description} with exactly the 60 keys of labels.json (same spelling, same order) to /home/alepom/progetti/lab-test-AP/classificatori/data/descriptions/massive_it_D.json (UTF-8, indent 1). Verify with a command that it parses and has 60 keys matching labels.json. Reply only with a 3-line summary (keys count, average words, labels you found genuinely ambiguous in training data).

---

## Message 2 to the agent

<system-reminder>
Your final report is delivered through SubagentHandback: when your work is complete, call SubagentHandback({message: <your full report>}). The call ends your run, so make it your last step. Only a SubagentHandback call reaches your caller as your result; plain text you write at the end is not delivered. Do not also send the report with SendMessage: your caller would get it twice.
</system-reminder>