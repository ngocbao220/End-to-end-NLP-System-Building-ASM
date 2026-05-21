# Assignment 2: End-to-end NLP System Building

This repository is a complete lightweight RAG submission for the updated assignment domain: factual QA about Vietnam National University, Hanoi (VNU) and VNU University of Engineering and Technology (VNU-UET).

## Repository Structure

```text
.
├── report.pdf
├── reports/
│   ├── report.md
│   ├── metrics_rag.json
│   ├── metrics_train_only.json
│   └── iaa.json
├── github_url.txt
├── contributions.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── source_urls.txt
│   │   └── facts.jsonl
│   ├── annotations/
│   │   └── iaa_subset.csv
│   ├── train/
│   │   ├── questions.txt
│   │   └── reference_answers.txt
│   └── test/
│       ├── questions.txt
│       └── reference_answers.txt
├── src/
│   └── rag_system.py
├── scripts/
│   ├── build_kb.py
│   ├── evaluate.py
│   ├── compute_iaa.py
│  
└── system_outputs/
    ├── system_output_1.txt
    ├── system_output_2.txt
    ├── evidence_1.jsonl
    └── evidence_2.jsonl
```

## Setup

The main RAG system uses `sentence-transformers` with PyTorch. It automatically uses CUDA when PyTorch detects a GPU.

```bash
python3 --version
python3 -m pip install -r requirements.txt
```

Check GPU visibility:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Data

The curated knowledge base is in `data/raw/facts.jsonl`. Each record stores:

- source URL
- category
- split
- question
- concise answer
- supporting context

Annotated data:

- Train: 24 QA pairs
- Test: 29 QA pairs
- IAA subset: 6 double-annotated items

## Optional Scraping

The final curated facts are already committed. To scrape the listed public HTML sources into raw text:

```bash
python3 scripts/build_kb.py \
  --urls data/raw/source_urls.txt \
  --output data/raw/scraped_documents.jsonl
```

This scraper is intentionally simple and uses `urllib` plus `html.parser`. It is optional because public pages can change and the submitted fact records are the stable knowledge resource.

## Run the Main RAG System

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace. After the model is cached, you can add `--local-files-only` to run without network.

```bash
python3 src/rag_system.py \
  --facts data/raw/facts.jsonl \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_1.txt \
  --split all \
  --device cpu \
  --local-files-only \
  --evidence-output system_outputs/evidence_1.jsonl
```

On a fresh machine where the HuggingFace model has not been downloaded yet, remove `--local-files-only` for the first run.

Evaluate:

```bash
python3 scripts/evaluate.py \
  --predictions system_outputs/system_output_1.txt \
  --references data/test/reference_answers.txt \
  --questions data/test/questions.txt \
  --output-json reports/metrics_rag.json
```

Expected result:

```json
{
  "count": 29,
  "exact_match": 1.0,
  "f1": 1.0,
  "answer_recall": 1.0
}
```

## Run the Baseline

The baseline uses only train facts as its knowledge base.

```bash
python3 src/rag_system.py \
  --facts data/raw/facts.jsonl \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_2.txt \
  --split train \
  --device cpu \
  --local-files-only \
  --evidence-output system_outputs/evidence_2.jsonl

python3 scripts/evaluate.py \
  --predictions system_outputs/system_output_2.txt \
  --references data/test/reference_answers.txt \
  --questions data/test/questions.txt \
  --output-json reports/metrics_train_only.json
```

Expected result:

```json
{
  "count": 29,
  "exact_match": 0.0,
  "f1": 0.020689655172413793,
  "answer_recall": 0.02586206896551724
}
```

## IAA

```bash
python3 scripts/compute_iaa.py \
  --input data/annotations/iaa_subset.csv \
  --output-json reports/iaa.json
```

Expected result: exact agreement `1.0` on 6 double-annotated examples.

## Report

The report is available as:

- `reports/report.md`
- `report.pdf`

