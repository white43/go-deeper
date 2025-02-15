package main

import (
	"fmt"
	gd "github.com/white43/go-deeper"
	"log"
)

func main() {
	n := gd.NewNetwork()
	n.AddLayer(gd.NewInputLayer(784))
	n.AddLayer(gd.NewHiddenLayer(30, gd.NewSigmoid()))
	n.AddLayer(gd.NewOutputLayer(10, gd.NewSoftmax()))

	n.SetOptimizer(gd.NewSGD(0.9, true))
	n.SetLossFunction(gd.NewCategoricalCrossEntropy(gd.ReductionMean))

	trainX, trainY, err := Load("train")
	if err != nil {
		log.Fatalln(err)
	}
	valX, valY, err := Load("t10k")
	if err != nil {
		log.Fatalln(err)
	}

	fmt.Printf("Loaded %d images in %d classes for training and %d for validation\n", len(trainX), len(trainY[0].RawMatrix().Data), len(valX))

	n.AddCallback(gd.NewSaveBest(gd.NewExporter(), 0.9))
	n.AddCallback(gd.NewEarlyStopping(25))

	lr := gd.NewFlatLearningRate(0.1)
	//lr := gd.NewCosineDecayLearningRate(0.1, 0.01)

	options := gd.FitOptions{
		TrainX:       trainX,
		TrainY:       trainY,
		ValX:         valX,
		ValY:         valY,
		Epochs:       10,
		BatchSize:    32,
		LearningRate: lr,
	}

	n.Fit(options)

	evaluation := n.Evaluate(trainX, trainY)
	fmt.Printf("Training Accuracy: %f\n", evaluation.Accuracy)
	println(evaluation.ConfusionMatrix())

	evaluation = n.Evaluate(valX, valY)
	fmt.Printf("Validation Accuracy: %f\n", evaluation.Accuracy)
	println(evaluation.ConfusionMatrix())
}
