package deeper

import (
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Activation interface {
	Activation(matrix *mat.Dense) *mat.Dense
	Derivative(matrix *mat.Dense) *mat.Dense
}

type Sigmoid struct{}

func NewSigmoid() Activation {
	return &Sigmoid{}
}

// Activation computes 1 / (1 + e^-n) for a vector
func (s Sigmoid) Activation(m *mat.Dense) *mat.Dense {
	if m.RawMatrix().Cols != 1 {
		log.Fatal("input with one column is expected")
	}

	vec := m.ColView(0)
	tmp := mat.NewDense(vec.Len(), 1, nil)

	tmp.Apply(func(_, _ int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	}, vec)

	return tmp
}

// Derivative computes σ(x) * (1 - σ(x)) which is the derivative of the
// sigmoid function. See https://math.stackexchange.com/a/1225116 for the
// derivation process
func (s Sigmoid) Derivative(m *mat.Dense) *mat.Dense {
	if m.RawMatrix().Cols != 1 {
		log.Fatal("input with one column is expected")
	}

	vec := m.ColView(0)
	tmp := mat.NewDense(vec.Len(), 1, nil)

	tmp.Apply(func(i, j int, v float64) float64 {
		return v * (1.0 - v)
	}, vec)

	return tmp
}

type Softmax struct{}

func NewSoftmax() Activation {
	return &Softmax{}
}

// Activation computes exp(x) / sum(exp(x)) for a vector
func (sm Softmax) Activation(m *mat.Dense) *mat.Dense {
	if m.RawMatrix().Cols != 1 {
		log.Fatal("input with one column is expected")
	}

	vec := m.ColView(0)
	tmp := mat.NewDense(vec.Len(), 1, nil)

	var sum float64

	for i := range vec.Len() {
		sum += math.Exp(vec.AtVec(i))
	}

	tmp.Apply(func(_, _ int, v float64) float64 {
		return math.Exp(v) / sum
	}, vec)

	return tmp
}

// Derivative computes and returns only main diagonal (i == j) of derivative
// of the softmax function for a vector.
//
// See the below links for information regarding the softmax function's
// derivative and its computation process with respect to indexes i and j.
// https://math.stackexchange.com/a/945918
// https://stats.stackexchange.com/a/453567
func (sm Softmax) Derivative(m *mat.Dense) *mat.Dense {
	if m.RawMatrix().Cols != 1 {
		log.Fatal("input with one column is expected")
	}

	vec := m.ColView(0)
	tmp := mat.NewDense(vec.Len(), 1, nil)

	tmp.Apply(func(_, _ int, v float64) float64 {
		return v * (1.0 - v) // Handling only the case when i == j
	}, vec)

	return tmp
}
