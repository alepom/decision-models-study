# Write Banking77 label descriptions

Model(s): claude-opus-5-5

## Message 1 to the agent

Task: write category descriptions ("condition D") for the 77 intents of the Banking77 dataset, for a research study on intent classification. They will be given to zero-shot classifiers together with the label names.

Use ONLY these two files (do not open any other file or directory; in particular never look for test data):
- /tmp/claude-1000/-home-alepom-progetti-lab-test-AP-classificatori/d1fea9b4-435b-421a-9137-8a40df42cf07/scratchpad/desc/banking77/labels.json  (label -> humanized name)
- /tmp/claude-1000/-home-alepom-progetti-lab-test-AP-classificatori/d1fea9b4-435b-421a-9137-8a40df42cf07/scratchpad/desc/banking77/train_full.jsonl  (training messages: {"text","label"})
You may also use your general knowledge of the public Banking77 dataset documentation (PolyAI), but do not search the web.

Method: for each label, read a sample of its training messages (e.g. 20-40, use shell tools like grep/shuf to sample), and especially compare labels that are easy to confuse (e.g. card_arrival vs card_delivery_estimate, pending_* vs declined_*, top_up_* variants, transfer_* variants, Refund_not_showing_up vs request_refund, cash_withdrawal_charge vs extra_charge_on_statement, etc.).

Write for each label ONE description in English, 1-3 short sentences (max ~45 words), containing: what the customer wants or reports (inclusion rule), and, where relevant, the boundary with the closest confusable labels ("not ...: use X"). Be factual about what the training data shows. No examples copied verbatim from the data.

Output: write a JSON object {label: description} with exactly the 77 keys of labels.json (same spelling, same order) to /home/alepom/progetti/lab-test-AP/classificatori/data/descriptions/banking77_D.json (UTF-8, indent 1). Then verify with a command that the file parses, has 77 keys matching labels.json, and report the longest description length in words. Reply only with a 3-line summary (keys count, average words, any label you found genuinely ambiguous in the training data).

---

## Message 2 to the agent

<system-reminder>
Your final report is delivered through SubagentHandback: when your work is complete, call SubagentHandback({message: <your full report>}). The call ends your run, so make it your last step. Only a SubagentHandback call reaches your caller as your result; plain text you write at the end is not delivered. Do not also send the report with SendMessage: your caller would get it twice.
</system-reminder>