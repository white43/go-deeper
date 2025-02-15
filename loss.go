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
	Reset()
	Loss(prediction, truth *mat.Dense) float64
	Derivative(prediction, truth *mat.Dense) *mat.Dense
	Result(count int) float64
}

type BinaryCrossEntropy struct {
	Sum       float64
	Reduction int
}

func NewBinaryCrossEntropy(reduction int) Loss {
	return &BinaryCrossEntropy{
		Reduction: reduction,
	}
}

func (bce *BinaryCrossEntropy) Reset() {
	bce.Sum = 0
}

func (bce *BinaryCrossEntropy) Result(count int) float64 {
	switch bce.Reduction {
	case ReductionMean:
		return bce.Sum / float64(count)
	case ReductionSum:
		return bce.Sum
	}

	return math.NaN()
}

func (bce *BinaryCrossEntropy) Loss(prediction, truth *mat.Dense) float64 {
	y := truth.RawMatrix().Data[0]
	yHat := prediction.RawMatrix().Data[0]

	return -(y*math.Log(yHat) + (1-y)*math.Log(1-yHat))
}

func (bce *BinaryCrossEntropy) Derivative(prediction, truth *mat.Dense) *mat.Dense {
	bce.Sum += bce.Loss(prediction, truth)

	tmp := mat.NewDense(truth.RawMatrix().Rows, truth.RawMatrix().Cols, nil)
	tmp.Sub(prediction, truth)

	return tmp
}

type CategoricalCrossEntropy struct {
	Sum       float64
	Reduction int
}

func NewCategoricalCrossEntropy(reduction int) Loss {
	return &CategoricalCrossEntropy{
		Reduction: reduction,
	}
}

func (cce *CategoricalCrossEntropy) Reset() {
	cce.Sum = 0
}

func (cce *CategoricalCrossEntropy) Result(count int) float64 {
	switch cce.Reduction {
	case ReductionMean:
		return cce.Sum / float64(count)
	case ReductionSum:
		return cce.Sum
	}

	return math.NaN()
}

// Loss computes Categorical cross-entropy. This operation is broken down into
// two steps: 1) we need to compute softmax function over the prediction vector
// 2) calculate entropy between predictions and truth vectors
func (cce *CategoricalCrossEntropy) Loss(prediction, truth *mat.Dense) float64 {
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

func (cce *CategoricalCrossEntropy) Derivative(prediction, truth *mat.Dense) *mat.Dense {
	cce.Sum += cce.Loss(prediction, truth)

	tmp := mat.NewDense(truth.RawMatrix().Rows, truth.RawMatrix().Cols, nil)
	tmp.Sub(prediction, truth)

	return tmp
}
