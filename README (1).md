# Text Similarity Analyzer — NLP Quiz 3

A Streamlit web app that uses the free pretrained model `sentence-transformers/all-MiniLM-L6-v2` to compute and visualize **text / sentence similarity** without any preprocessing, training, or dataset cleaning.

## Model Used
**`sentence-transformers/all-MiniLM-L6-v2`** — Free, open-source embedding model from HuggingFace. Generates 384-dimensional sentence embeddings and computes cosine similarity.

## App Purpose
- Accept user-entered sentences or words
- Compute similarity scores against a reference sentence
- Display results via three clear graphs
- Evaluate the analysis using Paul's Critical Thinking Standards (scored as %)

## Streamlit App Link
> _Add your Streamlit Community Cloud link here after deployment_

## Graphs Included
| Graph | Purpose |
|-------|---------|
| Bar Chart | Top similar sentences with exact cosine similarity scores |
| Heatmap | Pairwise similarity between all input sentences |
| Paul's Standards Score Chart | Critical thinking quality scored per standard (%) |

## Paul's Critical Thinking Standards Applied
| Standard | How it's applied |
|----------|-----------------|
| Clarity | Input word count and reference sentence are stated |
| Accuracy | Model name shown; no unsupported claims made |
| Precision | Exact decimal scores shown for every sentence |
| Relevance | All graphs directly support the similarity results |
| Logic | Top result explained using its score |
| Significance | Highest-scoring sentence is highlighted |
| Fairness | Lowest-scoring sentence noted as a model limitation |

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure
```
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Rules Followed
- No preprocessing (no tokenization, stopword removal, stemming, lemmatization)
- No model training or fine-tuning
- No paid API or paid model
- Free pretrained model only

## Screenshots
> _Add screenshots of your running app here_
