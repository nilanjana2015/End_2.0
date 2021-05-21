## Problem
Write a neural network that can:

Take 2 inputs:
 * an image from MNIST dataset, and
 * a random number between 0 and 9

Gives two outputs:
 * the "number" that was represented by the MNIST image, and
 * the "sum" of this number with the random number that was generated and sent as the input to the network
------------------------------------------------------------------------------------------------
## Approach

**Creation of Custom Dataset to include image labels and random numbers**
1. MNIST from torch datastes is downlaoded and splitted to train and test sets
```
  train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
```
2. Custom Data class is created to generate the data required for the problem. The __getitem__() and __len__() functions are implemented in the class. Each record in the MNIST dataset is appened with random number and sum of the label and the random number is considered as the output label 

Below code clip shows how the random numbers and MNIST dataset are appeneded in the class

```
class MNISTAndRandomBinaryNumbers(Dataset):
  
  def __init__(self, mnistDataset):
    self.data = []

    lengthOfDataset = len(mnistDataset)
    randomNumbers = torch.randint(0,9,(lengthOfDataset,))
    for i in range(lengthOfDataset):
      test = list(mnistDataset[i])
      test.append(self.getOneHotRepresentation(0,9,randomNumbers[i]))
      #test.append(self.getOneHotRepresentation(0,18,test[1]+randomNumbers[i]))
      test.append((torch.Tensor([test[1]+randomNumbers[i]])).type(torch.LongTensor))
      self.data.append(tuple(test))```
```

when we access the class we get 4 values, the first one is a image matrix and the second input is a one hot represented random number. The random number in the range of (0,9) is generated using the torch.randint method and its converted to one-hot vector representation. The other 2 are labels related to image class and random number 

**Neural Network**
## Building the Neural Network Model
The Model takes 2 inputs and generates 2 outputs and the model is trained on GPU. If GPU is not available the network will be switched to CPU device 

Below is the code to check if GPU is available or not

```
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
```


Below are the steps on how the network processes the data in different layers

In this design we pass the first input in layer1 and the second input is appended in the layer 3. We could pass in layer 1 but in this design its chosen to pass in layer 3. The convolution layers are generally used for averaging out the impact from neigbourhood pixels which best suits for images so we chose to pass the random number from layer 3 

### Layer 1
1. The first input which is an image of 1x28x28 is passed through a convolution layer with a kernelsize = 5 and out channels as 6. The output of this layer is 6 outputs / features of the size 1x24x24. 
2. The first layer output is passed Relu activation and through Max pooling layer of kernel size 2 and stride 2. Each feature 1x24x24 is max pooled to 1x12x12 size

### Layer 2 
3. The output of the layer 1 is taken as input to second convolution layer which has a kernel size of 5 and out channels as 12. The 6 input tensors of size 1x12x12 is converted to 12 features of size 1x8x8.
4. The output of the second layer convolution operation is passed to a Relu activation and to a max pool layer with kernel 2 and stride 2. Each feature of 1x12x12 is max pooled to size 1x4x4

### Layer 3.a
5. The second input is passed as input to this layer. This layer takes 1x10 as input and generates 1xsizeDefinedByuser (passed as parameter to the model, here it is 30) as the output
6. This output is passed through a Relu activation layer

### Layer 3.b 
7. The 12 features from output of layer 2 is flattened to [1, 192] tensor and the second input is concatenated to this output. This concatenated tensor acts as input to layer 3
8.The layer 3 takes 1x222 vector as input and generates 120 features as output
9. weighted input of layer 3.b neurons are passed through relu activation function

### Layer 4
10. The layer 3 takes 1x120 vector as input and generates 1x60 features as output
11. The weighted input of layer 4 neurons are passed through relu activation function

### Layer 5
12. The layer 4 takes 1x60 vector as input and generates 1x45 features as output
13. The weighted input of layer 5 neurons are passed through relu activation function

### Layer 6 - Output 1
14. The layer 5 takes 1x45 vector as input and generates 1x10 features as output
15. The output of the layer 6 is passed through softmax function to generate the probabilites of the class 

### Layer 7 - Output 2
16. The layer 5 takes 1x45 vector as input and generates 1x19 features as output
17. The output of the layer 7 is passed through softmax function to generate the probabilites of the class 


```
NetworkModel(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
  (fc0): Linear(in_features=10, out_features=30, bias=True)
  (fc1): Linear(in_features=222, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=60, bias=True)
  (fc3): Linear(in_features=60, out_features=45, bias=True)
  (out1): Linear(in_features=45, out_features=10, bias=True)
  (out2): Linear(in_features=45, out_features=19, bias=True)
)

```
## Training the Model

Below is the script for accessing the train dataset using torch DataLoader which gets data from the custom dataset in batch size of 32 

```
batchSize = 32
dataset = MNISTAndRandomBinaryNumbers(train_set)
train_loader = torch.utils.data.DataLoader(dataset
    ,batch_size=batchSize 
    ,shuffle=True
)
```
Following are the Hyperparameters that has been used for the training

1. Stochastic Gradient Descent Optimization technique has been used for training the model with learning rate of 0.001 and momentum 0.9

2. Cross entropy loss has been used for calculating the loss for both the outputs as the model generates probability distribution of the classes. Cross entropy generates high loss value when the model deviates from the actual prediction and generates small loss when the prediction is close.

3. The loss generated by both the outputs has been added to create a single loss and this has been used for gradient and back propagations. I have tried different by giving different weights to the losses but there is not much improvement in the model performance

4. Model is trained for 100 epochs. Under each epoch, the model is trained for every batch of 32 records and the weights are updated by calculating the loss at the end of each batch processing.

Below is the sreenshot of the training of the model

![image](https://user-images.githubusercontent.com/24980224/119090077-c4507f00-ba28-11eb-8e18-e9a2a2cff961.png)

Below is the decrease in the error with each epoch (learning rate =0.001)

![image](https://user-images.githubusercontent.com/24980224/119090165-e3e7a780-ba28-11eb-9394-6ef6c5172d93.png)


Below is the decrease in error with each epoch (learning rate =0.01)

![image](https://user-images.githubusercontent.com/24980224/119123786-93367580-ba4d-11eb-8f13-e87b8f646211.png)


### Testing the accuracy of the model

1. The test data split from MNIST has been used to create a new test dataset
2. The trained model is used for prediciting the labels of the images and also sum of the (image label and random number)
3. The predictions are evaluated against the original labels in the test dataset. 
4. Individual accuracies for image and random number sum are calculated by diving the right predictions with total records in the dataset
5. Combined accuracy where both are correctly predicted is also calculated 

Below is the accuracy of the model having a learning rate of 0.001
* The Combined accuracy of the model is 38.75%
* Image Label prediction accuracy is 98.35%
* Random Sum prediction accuracy is 38.97%


Below is the accuracy of the model having a learning rate of 0.01 while the rest all being same
* The Combined accuracy of the model is 54.8%
* Image Label prediction accuracy is 98.97%
* Random Sum prediction accuracy is 55.07%

### Improvements that can be made

The overall loss of the model is high and the accuracies are low, we can try the following methods to improve the model accuracies

1. Train for more number of epochs to further fit the model.
2. Increase the depth of the network by adding more additional layers
3. Increase the learning rate to increase the gradient updates
4. Image labels accuracies are very high, may be penalizing the features related to image or by providing additional weightage to the features related to randomsum
5. Switch the optimization from SGD to Adam may help in reducing the error or finding the optimal loss 



