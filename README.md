# Detection of Fake Reviews on Online Review Platforms using Deep Learning Architectures
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-data-augmentation/sentiment-analysis-on-yelp-fine-grained)](https://paperswithcode.com/sota/sentiment-analysis-on-yelp-fine-grained?p=unsupervised-data-augmentation)

Dataset: https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz <br>
         https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus

The data includes 1,569,264 samples from the Yelp Dataset Challenge 2015. This subset has 280,000 training samples and 19,000 test samples in each polarity.
The following implementations are done:
1. Bidirectional LSTM with GLoVE 50D word embeddings
2. LSTM with GLoVE 100D word embeddings
3. LSTM with GLoVE 300D word embeddings
4. CNN-LSTM with Doc2Vec and TF-IDF
5. Attention mechanism with GLoVe 100D word embeddings
6. Logistic Regression 
7. Multinomial Naive Bayes
8. Support Vector Machine - Stochastic Gradient Descent (SGD) 

The results obtained were as follows:


| Sr. No. | Model Accuracy (%) | Precision Score | Recall Score | F1 Score |
| ----- | ----------------- | ---------------- |-------------|------------|
| 1 | MultinomialNB | 90.25 | 0.9325 | 0.8601 | 0.8948 |
| 2 | Stochastic Gradient Descent (SGD) | 87.75 | 0.8913 | 0.8497 | 0.8700 |
| 3 | Logistic Regression | 87.00 | 0.8691 | 0.8601 | 0.8645 |
| 4 | Support Vector Machine | 56.25 | 0.525 | 0.9792 | 0.6835 |
| 5 | Gaussian Naive Bayes | 63.5 | 0.6424 | 0.6169 | 0.6294 |
| 6 | K-Nearest Neighbour | 57.5 | 0.8604 | 0.1840 | 0.3032 |
| 7 | Decision tree | 68.5 | 0.6681 | 0.7412 | 0.7028 |

| Model | Training accuracy(%) | Testing accuracy(%) |
| ----- | ----------------- | ---------------- |
| Bidirectional LSTM + GLoVe(50D) | 92.17  | 88.13 |
| LSTM + GLoVe(100D) | 99.18 | 85.75 |
| CNN + LSTM + Doc2Vec +TF-IDF | 96.23  | 92.19 |
| CNN + Attention + GLoVe(100D) | 99.00 | 90.25 |
| BiLSTM + Attention + GLoVe(100D) | 99.18 | 89.27 |
| CNN + BiLSTM + Attention + GLoVe(100D) | 99.75 | 81.25 |
| LogisticRegression + TF-IDF | 99.11 | 87.21 |

Future scope includes improvement in the attention layer to increase testing accuracy. BERT and XLNet can be implemented to improve the performance further.
