# Step 02 - Watch the LLM make things up (~10 min)

## What we're building
A script that asks the model questions about (a) a fictional company handbook
it has never seen and (b) fine details of the Attention paper. It then asks the
handbook question again with the text pasted into the prompt. Finally it works
out what pasting a *whole* PDF into every prompt would cost.

## Why this step exists
This is the problem RAG solves. LLMs don't know your private documents, and
when they half-know something they fill the gaps with confident, plausible
fiction. The dangerous part isn't that the model is sometimes wrong. It's that
right and wrong answers *look identical*. Then you'll see the fix in its
crudest form: put the facts in the prompt and the model uses them. The rest of
the workshop is about doing that automatically and efficiently, because you
can't paste 10,000 documents into every question.

## What to fill in
- **TODO 1** (`ask_without_context`): send just the question and return the answer text.
- **TODO 2** (`ask_with_context`): build a prompt of *instruction + context + question* and return the answer text.

## How to run it
```bash
python 02_hallucination/starter.py
python 02_hallucination/starter.py my.pdf -q "A detailed question about my.pdf"
```

## What success looks like
```
PART A - a private document the model has never seen
Without context:
  I'm sorry, I don't have access to Nimbus Analytics' handbook...   <- or a made-up number!
With the handbook pasted in:
  Employees can work remotely up to 3 days per week, and the internet
  reimbursement is up to INR 1,500 per month.

PART B - a famous paper the model half-remembers
Q: ... what dropout rate was used for the big Transformer model trained on English-to-French?
A: ... a dropout rate of 0.3 ...          <- wrong: the paper says 0.1 for EN-FR

PART C - why not just paste the whole document every time?
attention_is_all_you_need.pdf: 15 pages, 39,496 characters, ~9,874 tokens
```
Results vary between runs. Sometimes the model refuses politely and sometimes
it invents an answer. Run it twice and compare. Both outcomes make the point:
**without the source, you can't trust it and you can't check it.**

## Peek-ahead
TODO 2 is just an f-string: `f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}"`.
Still stuck? → [`solutions/02_hallucination.py`](../solutions/02_hallucination.py)
