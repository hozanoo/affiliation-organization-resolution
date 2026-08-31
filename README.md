# Affiliation Pipeline

A small Python pipeline for cleaning heterogeneous affiliation strings, extracting location information, and matching organizations against the [Research Organization Registry (ROR)](https://ror.org/).

The goal is not to resolve every affiliation automatically. Affiliation data is highly heterogeneous, so fully correct automated matching cannot be guaranteed. In tests on larger datasets, however, the pipeline showed strong matching consistency and proved useful for reducing manual work. Unmatched records are also expected, since not every organization is registered in ROR; their share therefore depends on the composition of the source data. The pipeline performs repetitive preprocessing and ROR matching, preserves uncertain cases for review, and produces structured output that can be checked and refined manually, for example in OpenRefine.

## Setup

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

The project uses packages such as `pandas`, `requests`, `tqdm`, `pycountry`, `Babel`, and IPython/Jupyter support.

## GeoNames data

City and alternate-name matching uses local GeoNames data. Download and extract:

- [`cities500.zip`](https://download.geonames.org/export/dump/cities500.zip)
- [`alternateNamesV2.zip`](https://download.geonames.org/export/dump/alternateNamesV2.zip)

Place the extracted files here:

```text
data/
├── cities500/
│   └── cities500.txt
└── alternateNamesV2/
    └── alternateNamesV2.txt
```

These large GeoNames files are intentionally not stored in the repository.

## ROR API Client ID

ROR states that from Q3 2026 a valid Client ID is required to receive the standard rate limit of **2,000 requests per 5 minutes**. Requests without identification receive a lower limit of **50 requests per 5 minutes**.

Register for a free Client ID:

https://ror.org/api-client-id

Create a local `.env` file in the project root:

```text
ROR_CLIENT_ID=YOUR_CLIENT_ID
```

The `.env` file is ignored by Git. ROR Client IDs are used for client identification rather than authentication, but keeping local configuration outside the repository keeps the project setup clean.

ROR documentation:

https://ror.readme.io/docs/client-id

## Basic workflow

```text
raw CSV
  ↓
preprocessing / affiliation splitting
  ↓
location extraction and affiliation_clean
  ↓
ROR affiliation matching + exact-name fallback
  ↓
structured affiliation_clean_1, affiliation_clean_2, ...
  ↓
matched / unmatched / review
  ↓
manual quality control
```

By default, the tutorial uses the included public demo dataset:

data/affilation_test.csv

This file contains synthetic/demo affiliation examples based on publicly known organization names and is intended for testing the pipeline.

To use another dataset, copy your CSV into `data/` and change `DATA_FILENAME` and, if necessary, `SOURCE_AFFILIATION_COLUMN` in the configuration cell.

## Project structure

```text
affiliation-pipeline/
├── affiliation_pipeline/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── locations.py
│   └── ror_api.py
├── data/
├── jupyter/
│   ├── tutorial_affiliation_pipeline_v1.ipynb
│   └── tutorial_affiliation_pipeline_en_v1.ipynb
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Tutorials

The notebooks contain the full workflow, function explanations, optional parameters, examples, status handling, and instructions for using your own data:

- [English tutorial](jupyter/tutorial_affiliation_pipeline_en_v1.ipynb)
- [German tutorial](jupyter/tutorial_affiliation_pipeline_v1.ipynb)

For first-time use, start with one of these notebooks.
