go-deeper
---------

This toy library is an attempt to wrap my head around neural networks and their
inner processes starting from relatively simple concepts and going **deeper**
gradually. Its naive implementation is a result of self-directed learning from
books, examples, and university subjects. This library is in a very early state,
so anything may (and will) change along the development process, since new
features will almost certainly require refactoring. It is designed to use
interfaces extensively which makes it (hopefully) extensible.

Current features
----------------

* Feedforward and backpropagation
* Optimizers: `SGD` (with Nesterov Accelerated Gradient)
* Activation functions: `Sigmoid`, `SoftMax`
* Loss functions: `CategoricalCrossEntropy`
* Learning rate schedulers: `Flat`, `Cosine decay`
* Callbacks: `Early stopping`, `Save best model`
* Export: save to dsk and load saved weights

How to use it
-------------

```go
import gd "github.com/white43/go-deeper"

// Create a new network and its layers
n := gd.NewNetwork()
n.AddLayer(gd.NewInputLayer(784))
n.AddLayer(gd.NewHiddenLayer(30, gd.NewSigmoid()))
n.AddLayer(gd.NewOutputLayer(10, gd.NewSoftmax()))

// Define the way its weights will be adjusted
// In this example, Stochastic Gradient Descent with Nesterov Accelerated Gradient is used
n.SetOptimizer(gd.NewSGD(0.9, true))
// Categorical cross entropy for multiclass classification
n.SetLossFunction(gd.NewCategoricalCrossEntropy(gd.ReductionMean))

// Create data for training and validation
// This data is expected to be in the form of *mat.Dense slices
// See https://github.com/gonum/gonum
trainX, trainY, _ := Load("train")
valX, valY, _ := Load("t10k")

// Set learning rate
lr := gd.NewFlatLearningRate(0.1)

options := gd.FitOptions{
    TrainX:       trainX,
    TrainY:       trainY,
    ValX:         valX,
    ValY:         valY,
    Epochs:       10,
    BatchSize:    32,
    LearningRate: lr,
}

// Train the network    
evaluation := n.Fit(options)

// Print model classification statistics
fmt.Printf("Validation Accuracy: %f\n", evaluation.Accuracy)
println(evaluation.ConfusionMatrix())
```

TODO
----

* [ ] Prediction of individual and batches of samples 
* [ ] Arbitrary number of channels (colours)
* [ ] Binary cross entropy loss function
* [ ] Adam/AdamW optimizer
* [ ] Lion optimizer
* [ ] Dropouts
* [ ] Batch normalization
* [ ] L1/L2 regularization
* [ ] Convolutional layers
* [ ] Max/global average pooling
* [ ] ReLU activation function

