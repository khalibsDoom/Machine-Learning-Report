# Comparing Classification Algorithms: Spam and Fake News Detection

A machine learning study comparing how different classification algorithms perform on two text classification problems: SMS spam detection and fake news detection.

## What it does

- **Text preprocessing:** lowercasing, tokenization, stop-word removal and Porter stemming with NLTK; word clouds and frequency analysis to explore each class
- **Feature extraction:** bag-of-words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`)
- **Model comparison:** trains and scores 10+ classifiers side by side on accuracy and precision: Gaussian, Multinomial and Bernoulli Naive Bayes, Logistic Regression, SVC, K-Nearest Neighbours, Decision Tree, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and a Passive Aggressive classifier for fake news
- **Evaluation:** confusion matrices, accuracy and precision, and a written analysis of why each algorithm behaves differently on each dataset

The full write-up, with results and conclusions, is in [`docs/analysis-report.pdf`](docs/analysis-report.pdf).

## Datasets

| Dataset | Source |
|---|---|
| SMS Spam Collection (`spam.csv`) | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Fake news (`news.csv`) | DataFlair fake news dataset |

Both files are included in this repo.

## Run it

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
python "Assignment 2.py"
```

## Tech stack

Python · scikit-learn · NLTK · pandas · NumPy · Matplotlib · Seaborn · WordCloud
