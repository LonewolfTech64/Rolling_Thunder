# Rolling_Thunder

Rolling Thunder Intelligence Analysis
A semantic‑retrieval system for analysing declassified CIA, JCS, SNIE and OSD documents from the Vietnam War (1965–1968).

Overview
This project applies modern NLP methods to a curated corpus of declassified U.S. intelligence and military documents relating to the Rolling Thunder bombing campaign.
The goal is to compare how different institutions — CIA, Joint Chiefs of Staff (JCS), Special National Intelligence Estimates (SNIE), and Office of the Secretary of Defense (OSD) — assessed:

bombing effectiveness

coercive leverage

infiltration resilience

escalation risk

political will of North Vietnam

The project combines semantic search, FAISS vector indexing, and transformer embeddings to support structured historical analysis.

Key Features
Semantic Retrieval Pipeline  
Built using FAISS and BGE‑Large‑EN embeddings for high‑quality document search.

Structured Corpus  
700+ declassified documents annotated with metadata (source, date, institutional bias, operational bias, rhetorical framing).

Comparative Analysis Tools  
Scripts to contrast CIA skepticism with JCS escalation advocacy, and to evaluate SNIE caution vs OSD cost‑effectiveness assessments.

RAG‑based Question Answering  
Retrieval‑augmented generation for synthesising multi‑document answers to strategic questions.

Historical Insight via ML  
Demonstrates how modern NLP can support historiography, strategic‑studies research, and intelligence analysis.

Example Queries
python
rag_answer("How did the CIA view the bombing of North Vietnam?")
rag_answer("How did the JCS view the bombing campaign?")
rag_answer("Compare CIA and JCS assessments of Rolling Thunder.")
rag_answer("Summarise SNIE conclusions about coercive leverage.")
Technical Stack
Python

FAISS

BGE‑Large‑EN embeddings

HuggingFace Transformers

NumPy / Pandas

Jupyter Notebooks

Project Motivation
This project sits at the intersection of:

machine learning

strategic studies

Cold War historiography

intelligence analysis

It demonstrates how modern NLP techniques can illuminate doctrinal divergence between U.S. institutions during the Vietnam War.

Repository Contents
src/ — embedding pipeline, retrieval system, RAG logic

index/ — FAISS index, embeddings, lookup tables

data/ — declassified corpus

notebooks/ — exploratory analysis and comparative studies

analysis/ — scripts for institutional comparison

Future Work
Add cross‑encoder reranking

Expand metadata annotation

Add visualisations of doctrinal divergence

Publish a working paper on institutional assessments of Rolling Thunder

Author
Jonathan F. D. Addison (Dale)
Data Analyst / Data Scientist
Sunderland, UK
