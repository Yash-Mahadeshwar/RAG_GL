# Sample documents

The PDFs are **not** committed to this repo (copyright, and repo size). Download
them here before the workshop. Every step defaults to the Attention paper, so
that one is the must-have.

| Save as | Source | How |
|---|---|---|
| `attention_is_all_you_need.pdf` | Vaswani et al., 2017 - [arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762) | Direct PDF download |
| `python_tutorial_intro.pdf` | [docs.python.org/3/tutorial/index.html](https://docs.python.org/3/tutorial/index.html) | Open in a browser → Print → **Save as PDF** |
| `how_to_do_great_work.pdf` | Paul Graham - [paulgraham.com/greatwork.html](https://paulgraham.com/greatwork.html) | Open in a browser → Print → **Save as PDF** |

## One-liner for the Attention paper

Run from the repo root:

```bash
# macOS / Linux
curl -L -o sample_docs/attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762

# Windows (PowerShell)
Invoke-WebRequest https://arxiv.org/pdf/1706.03762 -OutFile sample_docs/attention_is_all_you_need.pdf
```

If you download it in a browser it will be called `1706.03762.pdf`. Rename it
to `attention_is_all_you_need.pdf`.

## Using your own PDF

Every script accepts a PDF path in two ways:

```bash
# 1. as a command-line argument (works for one run)
python 04_naive_rag/starter.py sample_docs/how_to_do_great_work.pdf

# 2. as an environment variable (works for every step); add this to your .env
RAG_PDF_PATH=sample_docs/how_to_do_great_work.pdf
```

Text-based PDFs work best. Scanned PDFs, which are just images of pages, have
no extractable text, and every chunk will come out empty.

Heads-up: the demo questions in steps 02, 05 and 06 are written for the
Attention paper. With your own PDF, pass your own questions with
`-q "your question"`. In step 06, write your own eval set.
