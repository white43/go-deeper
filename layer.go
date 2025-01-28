package deeper

import (
	"gonum.org/v1/gonum/mat"
)

type BackpropagationLayer interface {
	Rows() int
	Cols() int
	IsInput() bool
	IsOutput() bool
	Weights() *mat.Dense
	Biases() *mat.Dense
	SetWeights(w *mat.Dense)
	SetBiases(b *mat.Dense)
	SetInput(l BackpropagationLayer)
	SetOutput(l BackpropagationLayer)
	Feedforward(x *mat.Dense, activations *Stack) *mat.Dense
	Backpropagation(delta *mat.Dense, activations, deltaWs, deltaBs *Stack)
}

type Layer struct {
	WeightInitializer
	id         int
	rows       int
	input      BackpropagationLayer
	output     BackpropagationLayer
	weights    *mat.Dense
	biases     *mat.Dense
	activation Activation
	isInput    bool
	isOutput   bool
}

func NewInputLayer(neurons int) BackpropagationLayer {
	return &Layer{
		rows:    neurons,
		isInput: true,
	}
}

func NewHiddenLayer(neurons int, activation Activation) BackpropagationLayer {
	return &Layer{
		rows:       neurons,
		activation: activation,
	}
}

func NewOutputLayer(neurons int, activation Activation) BackpropagationLayer {
	return &Layer{
		rows:       neurons,
		activation: activation,
		isOutput:   true,
	}
}

func (l *Layer) Rows() int {
	return l.rows
}

func (l *Layer) Cols() int {
	if l.isInput {
		return 1
	}

	return l.input.Rows()
}

func (l *Layer) IsInput() bool {
	return l.isInput
}

func (l *Layer) IsOutput() bool {
	return l.isOutput
}

func (l *Layer) Weights() *mat.Dense {
	return l.weights
}

func (l *Layer) Biases() *mat.Dense {
	return l.biases
}

func (l *Layer) SetWeights(w *mat.Dense) {
	l.weights = w
}

func (l *Layer) SetBiases(b *mat.Dense) {
	l.biases = b
}

func (l *Layer) SetInput(input BackpropagationLayer) {
	l.input = input
}

func (l *Layer) SetOutput(output BackpropagationLayer) {
	l.output = output
}

// Feedforward recursively passes input (x) through every layer it is connected.
// On every layer the following operations are being carried out: 1) y = wx + b,
// where "w" and "b" are weights and bias, respectively 2) activation(y) that
// returns a vector which is a result of activation function for this layer.
// This vector is input for the next layer. The activations is a stack that holds
// these vectors for every layer. They are used during backpropagation to
// calculate updates for weights and biases.
func (l *Layer) Feedforward(x *mat.Dense, activations *Stack) *mat.Dense {
	if !l.isInput {
		// y = wx + b
		y := mat.NewDense(l.weights.RawMatrix().Rows, 1, nil)
		y.Mul(l.weights, x)
		y.Add(y, l.biases)
		// x = activation(y)
		x = l.activation.Activation(y)
	}

	if activations != nil {
		activations.Push(x)
	}

	if l.output != nil {
		return l.output.Feedforward(x, activations)
	}

	return x
}

// Backpropagation recursively passes error calculated on the output level
// (delta) through every layer it is connected. The error is calculated using
// the activation function's derivative on the output layer (Loss.Derivative).
// On every layer we use output of the activation function stored in the
// activation stack to get updates to weights and bias. These weight updates
// placed in two stacks (deltaWs and deltaBs) for further subtraction.
func (l *Layer) Backpropagation(delta *mat.Dense, activations, deltaWs, deltaBs *Stack) {
	if l.isInput {
		return
	}

	deltaW := mat.NewDense(l.weights.RawMatrix().Rows, l.input.Rows(), nil)
	deltaB := mat.NewDense(l.biases.RawMatrix().Rows, 1, nil)

	// On output layer we use error (delta), while on intermediate layers we
	// use delta computed on previous step of the backpropagation process.
	if !l.isOutput {
		weightsT := l.output.Weights().T() // Transpose
		rows, _ := weightsT.Dims()
		cols := delta.RawMatrix().Cols

		// wᵀ * delta
		tmp := mat.NewDense(rows, cols, nil)
		tmp.Mul(weightsT, delta)
		delta = tmp
	}

	// delta ⊙ activation(x)' (Hadamard product)
	delta.MulElem(delta, l.activation.Derivative(activations.Pop()))
	deltaB.Copy(delta)

	// delta * activation(x)ᵀ
	deltaW.Mul(delta, activations.Peek().T()) // Transpose

	// Last layers go first to the stack to be on its bottom after recursion
	deltaWs.Push(deltaW)
	deltaBs.Push(deltaB)

	if l.input != nil {
		l.input.Backpropagation(delta, activations, deltaWs, deltaBs)
	}
}
