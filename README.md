# BERT_multilabel_text_classification
This project demonstrates how to make useof BERT enoder to train a multi label text classification problem. I have used the popular toxic comment classsifcation dataset from Kaggle.
This project makes use of [Bert-as-a-service](https://github.com/hanxiao/bert-as-service) project.

## Requirements
1. Python >= 3.5
2. TensorFlow >= 1.10
3. Keras
4. bert-serving-server and  bert-serving-client

## steps

1. Install bert-serving-server and bert-serving-client using the below commands
```
pip install bert-serving-server 
pip install bert-serving-client
```
2. Download the bert base uncased model from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) or using the below command
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && unzip uncased_L-12_H-768_A-12.zip
```
3. Start the bert-serving-server 
```
bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len 50
```
 Note: If you pass the max_seq_len parameter as None, the service will find the max length from each batch of text and use it for padding/truncating
 
4. Download the toxic text classification dataset from [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

5. Run the training script
```
python train.py

```
bert-serving-client objcet will bind to the server and encode the list of text which is being passed, into a (n, 768) dimension vector. Here n is the number of input sentences.

On top of this encoded vector we add a couple of dense layers to get the prediction.

## Loss function and accuracy metric
I have used sigmoid + focal loss as loss function instead of usual sigmoid + binary crross entropy to tackle the class imbalnce problem in multi label predictions
Also, instead of simple accuracy, AUCis being used. 

## Result
Result will be updated soon.


## References
1. [Google's Bert project](https://github.com/google-research/bert)
2. [Bert as a service](https://github.com/hanxiao/bert-as-service)
3. [Toxic comment challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

