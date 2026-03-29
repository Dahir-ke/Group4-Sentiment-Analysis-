# Advanced Sentiment Analysis for Apple & Google

This repository contains a Jupyter Notebook implementing an advanced Natural Language Processing (NLP) pipeline to automatically classify tweets about Apple and Google products into three sentiment categories: **Negative emotion**, **Positive emotion**, and **No emotion toward brand or product**. The project focuses on maximizing **recall for the negative class** to ensure timely detection of customer complaints and brand crises.

## Business Problem
High‑profile events like the SXSW conference generate thousands of tweets per hour. Manual monitoring is impossible, yet missing a negative tweet can lead to serious brand damage. The goal is to build an automated classifier that catches **at least 80%** of negative tweets, even if some neutral or positive tweets are mistakenly flagged.

## Dataset
The dataset (`judge_1377884607_tweet_product_company.csv`) contains tweets collected during the SXSW conference. It includes:
- `tweet_text`: raw tweet text
- `is_there_an_emotion_directed_at_a_brand_or_product`: sentiment label
- `emotion_in_tweet_is_directed_at`: product mentioned (Apple/Google)

### Class Distribution
- No emotion: ~59%
- Positive emotion: ~33%
- Negative emotion: ~6%
- “I can't tell” (removed)

## Methodology
1. **Text Preprocessing**  
   - Lowercasing, punctuation removal, URL/mention removal  
   - Stopword removal (except negations like “not”, “no”)  
   - Lemmatization using NLTK

2. **Feature Engineering**  
   - TF‑IDF vectorization with unigrams and bigrams, max_features=20,000, min_df=3, max_df=0.85

3. **Model Training & Comparison**  
   - Logistic Regression (with class_weight='balanced')  
   - Random Forest (class_weight='balanced')  
   - Support Vector Machine (LinearSVC) with custom class weights

4. **Threshold Tuning**  
   - Logistic Regression gave the highest base recall for negative class (0.51).  
   - Lowered the decision threshold for negative class from 0.5 to 0.15, boosting recall to **0.81**.

5. **Final Model – CrisisRecallModel**  
   - A custom Python class encapsulating TF‑IDF + Logistic Regression + threshold logic for easy deployment.

6. **Evaluation**  
   - Classification report, confusion matrix, false‑negative analysis, 5‑fold cross‑validation.

## Results

| Model                                    | Negative Recall | Accuracy |
|------------------------------------------|-----------------|----------|
| Logistic Regression (base)               | 0.51            | 0.63     |
| Random Forest                            | 0.20            | 0.67     |
| SVM                                      | 0.39            | 0.68     |
| **CrisisRecallModel (threshold=0.15)**   | **0.81**        | 0.47     |

The final model captures **81%** of all negative tweets, meeting the business requirement. Precision for negative class is low (0.13), but this trade‑off is acceptable because the cost of a missed negative tweet far outweighs the cost of reviewing false alarms.

## How to Run the Notebook

### Requirements
- Python 3.8 or higher
- Jupyter Notebook / JupyterLab/vscode

\