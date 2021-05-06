## Scope

Please find the answers for the first assignment questions in the below sections.

### What is a neural network neuron?
  Neuron is a mathematical operation takes the input and multiply by it's weight and passes the sum to another neuron via activation function.

### What is the use of the learning rate?
The learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value.

The learning rate controls how quickly the model is adapted to the problem. A smaller learning rate requires many updates before reaching the minimum point. Too large of a learning rate causes drastic updates which lead to divergent behaviors. So it is always important to choose optimal learning rate swiftly reaches the minimum point.

### How are weights intialized?
We can initialize the weights for neural network with differnt techinques such as Zero initialization, Random initialization, He initialization, Xavier initialization based on our network. The selected weight initialization technique should not affect the training purpose.

### What is "loss" in neural network?
  "Loss" is nothing but a prediction error. This is the difference between the expected output and predicted output.

### What is "chain rule" in gradient flow?
  The algorithm used to update the model parameters(weights, biases) in order to effectively train a neural network is known as chain rule.  
  Mathematically total output gradient is the total gradient caused by the all the neurons which are contributed for a output:  
    
  FinalGradient = GradientContribution**1** + GradientContribution**2**+ ....+ GradientContribution**N**
   
  Gradient<sub>i</sub> = GradientInside × GradientContribution<sub>i</sub> <!--∂Output∂wi=∂Contribution1∂wi×∂Output∂Contribution1 -->  
   
  <img align="center" src="https://render.githubusercontent.com/render/math?math=\frac{\partial _{Output}}{\partial _{w^i}} = \frac{\partial _{Contribution^i}}{\partial _{w^i}}    \times \frac{\partial _{Output}}{\partial _{Contribution^i}} ">
   
