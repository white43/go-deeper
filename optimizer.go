package deeper

import "gonum.org/v1/gonum/mat"

type Optimizer interface {
	Apply(weights, deltaWs *mat.Dense, lr float64)
}

type SGD struct{}

func NewSGD() Optimizer {
	return &SGD{}
}

func (s *SGD) Apply(weights, deltaWs *mat.Dense, lr float64) {
	tmp := mat.NewDense(deltaWs.RawMatrix().Rows, deltaWs.RawMatrix().Cols, nil)
	tmp.Scale(lr, deltaWs)
	weights.Sub(weights, tmp)
}
