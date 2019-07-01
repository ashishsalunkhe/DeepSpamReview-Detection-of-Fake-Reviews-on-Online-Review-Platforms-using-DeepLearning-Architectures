# Fraudulent-Review-Detection-using-CNN-LSTM-and-Word-Embeddings
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-data-augmentation/sentiment-analysis-on-yelp-fine-grained)](https://paperswithcode.com/sota/sentiment-analysis-on-yelp-fine-grained?p=unsupervised-data-augmentation)

Dataset: https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz

The data includes 1,569,264 samples from the Yelp Dataset Challenge 2015. This subset has 280,000 training samples and 19,000 test samples in each polarity.
The following implementations are done:
1. Bidirectional LSTM with GLoVE 50D word embeddings
2. LSTM with GLoVE 100D word embeddings
3. LSTM with GLoVE 300D word embeddings
4. Multi layer LSTM + with GLoVE 100D word embeddings
5. Attention mechanism with GLoVe 100D word embeddings

The results obtained were as follows:

| Model | Training accuracy(%) | Testing accuracy(%) |
| ----- | ----------------- | ---------------- |
| Bidirectional LSTM + GLoVe(50D) | 92.17  | 88.13 |
| LSTM + GLoVe(100D) |  |  |
| LSTM + GLoVe(300D) |  |  |
| Multi layer LSTM + GLoVe(300D) |  |  |
| LSTM + Attention + GLoVe(100D) |  | |

Future scope includes improvement in the attention layer to increase testing accuracy. BERT and XLNet can be implemented to improve the performance further.
