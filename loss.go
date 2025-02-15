package deeper

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

const (
	ReductionMean = iota
	ReductionSum
)

type Loss interface {
	Reset(batchSize int)
	Loss(prediction, truth *mat.Dense) float64
	Derivative(prediction, truth *mat.Dense) *mat.Dense
	Result() float64
}

type CategoricalCrossEntropy struct {
	Sum       float64
	BatchSize int
	Reduction int
}

func NewCategoricalCrossEntropy(reduction int) Loss {
	return &CategoricalCrossEntropy{
		Reduction: reduction,
	}
}

func (c *CategoricalCrossEntropy) Reset(batchSize int) {
	c.Sum = 0
	c.BatchSize = batchSize
}

func (c *CategoricalCrossEntropy) Result() float64 {
	switch c.Reduction {
	case ReductionMean:
		return c.Sum / float64(c.BatchSize)
	case ReductionSum:
		return c.Sum
	}

	return math.NaN()
}

// Loss computes Categorical cross-entropy. This operation is broken down into
// two steps: 1) we need to compute softmax function over the prediction vector
// 2) calculate entropy between predictions and truth vectors
func (c *CategoricalCrossEntropy) Loss(prediction, truth *mat.Dense) float64 {
	y := truth.RawMatrix().Data
	yHat := prediction.RawMatrix().Data

	sum := float64(0)

	for i := range len(yHat) {
		sum += math.Exp(yHat[i])
	}

	entropy := float64(0)

	for i := range len(y) {
		if y[i] > 0 {
			entropy += y[i] * math.Log(math.Exp(yHat[i])/sum)
		}
	}

	return -entropy
}

func (c *CategoricalCrossEntropy) Derivative(prediction, truth *mat.Dense) *mat.Dense {
	c.Sum += c.Loss(prediction, truth)

	tmp := mat.NewDense(truth.RawMatrix().Rows, truth.RawMatrix().Cols, nil)
	tmp.Sub(prediction, truth)

	return tmp
}
