# Step 06 - Evaluate your RAG pipeline (~20 min)

## What we're building
A small evaluation harness. A hand-written eval set of question / reference
answer / evidence-phrase triples is used to measure:
1. **Retrieval hit-rate@k**: for what fraction of questions does the chunk
   containing the evidence appear in the top k? It's compared for vector-only
   vs. vector + reranker at k = 1, 3, 5.
2. **Answer accuracy**: the full pipeline answers each question, and an
   **LLM-as-judge** grades it CORRECT or INCORRECT against the reference.

## Why this step exists
So far you've judged every change by eyeballing a few answers. That doesn't
scale, and it's how people convince themselves a change helped when it
didn't. Even 8 labelled questions turn "seems better" into a number you can
compare. Measuring retrieval and generation *separately* is the key habit. If
hit-rate is low, fix chunking or retrieval. If hit-rate is high but accuracy
is low, fix the prompt or model. If both are high and users still complain,
your eval set doesn't look like real questions.

## What to fill in
- **TODO 1** (`is_hit`): is the evidence phrase in any retrieved chunk? (Lowercase it and collapse whitespace.)
- **TODO 2** (`hit_rate`): the fraction of *answerable* questions with a hit in the top k.
- **TODO 3** (`judge`): invoke the judge chain and turn "CORRECT"/"INCORRECT" into a bool.
- **TODO 4** (`EVAL_SET`): add at least two QA pairs of your own from the PDF.

Note: the first error you'll see is TODO 2, because `hit_rate` runs before
`is_hit` is ever called. Do TODO 1 and TODO 2 together.

## How to run it
```bash
python 06_evaluation/starter.py
```
It makes ~20 LLM calls (an answer and a judgement per question), which takes
about 30 seconds.

## What success looks like
```
Retrieval hit-rate on 8 answerable questions
    k   vector only   + rerank
    1           62%        88%
    3           88%       100%
    5          100%       100%

Answer accuracy (reranked top 4, judged by gpt-4o-mini)
  PASS  How many identical layers are in the Transformer encode -> The Transformer encoder consists of 6 identical layers [p. 3].
  PASS  How many attention heads does the base Transformer use? -> The base Transformer uses 8 attention heads [p. 5].
  ...
  PASS  Who won the 2018 FIFA World Cup?                        -> I don't know based on the provided document.

  Accuracy: 9/9 = 100%
```
Your numbers will differ. Expect the reranker to help most at k = 1. If you
score 100% on everything, the eval set is too easy, and that's a finding
too. Write a harder question.

**Discuss:**
- The judge is an LLM, the same model as the system being judged. When could it
  be wrong? (Hint: try a reference answer of "6" and a system answer of "six layers".)
- The evidence check is a substring match. What happens if two chunks contain
  "2048" but only one answers the question?
- How would you build an eval set for a real product? (Answer: from real user
  questions, labelled by people who know the domain.)

## Peek-ahead
`hit_rate` is `sum(is_hit(retrieve(q, k), ev) for ...) / len(answerable)`.
Booleans add up as 0/1 in Python.
Still stuck? → [`solutions/06_evaluation.py`](../solutions/06_evaluation.py)
