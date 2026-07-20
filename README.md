# BBC News Unsupervised Topic Modeling (LDA vs NMF)

## Project Overview
This project applies unsupervised topic modeling to BBC news articles to uncover latent themes and compare two widely used approaches: Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF). The workflow includes text preprocessing, token cleaning, topic coherence tuning, model training, and interactive topic visualization.

## Business Objective
The goal is to turn a large collection of news content into structured insights that help identify recurring topics, monitor narrative themes, and support downstream NLP or content analytics workflows.

## Key Features
- Linguistic preprocessing and text cleaning for news articles
- Token-based document representation suitable for topic modeling
- Coherence-based tuning for selecting an appropriate number of topics
- Comparative LDA vs NMF topic extraction
- Interactive topic exploration using pyLDAvis
- Exportable visual artifacts such as coherence curves and topic dashboards

## Installation
Create and activate a Python environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the full pipeline from the project root:

```bash
python main.py
```

This will:
1. Load the BBC news dataset
2. Preprocess the text data
3. Train topic models
4. Export reports and visualizations to the reports folder

## Visual Results
The repository includes a coherence curve report here:

![Coherence Score Curve](reports/coherence_score_curve.png)
