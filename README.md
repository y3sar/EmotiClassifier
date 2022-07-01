![Alt text](./assets/EmotiClassifier.png)
# EmotiClassifier

A simple CNN model built to classify emotions. This model was trained on FER 2013 dataset
The model has a validation accuracy of 64% which is very far from the SOTA 76%

# Notebook
The jupyter notebook also contains an EfficientNet and a Resnet applied on the emotion classificaiton problem. Using the pretrianed models they did not produce significant performance. This is because the FER 2013 dataset contains grayscale images and these models take RGB images. So changing the first layer of the models to take an input of 1 channel significantly reduced performance


## Usage

```shell
$ python app.py
```

