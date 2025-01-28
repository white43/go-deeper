package deeper

import (
	"math"
	"slices"

	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Loss(prediction, truth *mat.Dense) float64
	Derivative(prediction, truth *mat.Dense) *mat.Dense
}

type CategoricalCrossEntropy struct{}

func NewCategoricalCrossEntropy() Loss {
	return &CategoricalCrossEntropy{}
}

func (c *CategoricalCrossEntropy) Loss(prediction, truth *mat.Dense) float64 {
	logSoftMax := mat.NewDense(prediction.RawMatrix().Rows, prediction.RawMatrix().Cols, nil)
	logSoftMax.Copy(prediction)

	// max_x = np.max(x, axis=axis, keepdims=True)
	maxPred := slices.Max(logSoftMax.RawMatrix().Data)

	// logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
	logSumExp := float64(0)

	for i := range logSoftMax.RawMatrix().Data {
		logSumExp += math.Exp(logSoftMax.RawMatrix().Data[i] - maxPred)
	}

	logSumExp = math.Log(logSumExp)

	// log_softmax = x - max_x - logsumexp
	for i := range logSoftMax.RawMatrix().Data {
		logSoftMax.RawMatrix().Data[i] -= (maxPred + logSumExp)
	}

	// Hadamard product
	logSoftMax.MulElem(prediction, logSoftMax)

	sum := float64(0)

	for i := range logSoftMax.RawMatrix().Data {
		sum -= truth.RawMatrix().Data[i] * logSoftMax.RawMatrix().Data[i]
	}

	return sum
}

func (c *CategoricalCrossEntropy) Derivative(prediction, truth *mat.Dense) *mat.Dense {
	tmp := mat.NewDense(truth.RawMatrix().Rows, truth.RawMatrix().Cols, nil)
	tmp.Sub(prediction, truth)

	return tmp
}
