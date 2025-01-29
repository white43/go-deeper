package deeper

import (
	"gonum.org/v1/gonum/mat"
	"slices"
	"unsafe"
)

type Optimizer interface {
	Apply(weights, deltaWs *mat.Dense, lr float64)
}

type SGD struct {
	momentum  float64
	momentums map[uintptr]*mat.Dense
}

func NewSGD(momentum float64) Optimizer {
	return &SGD{
		momentum:  momentum,
		momentums: make(map[uintptr]*mat.Dense),
	}
}

func (s *SGD) Apply(weights, deltaWs *mat.Dense, lr float64) {
	if s.momentum > 0 && s.momentum < 1 {
		p := uintptr(unsafe.Pointer(&weights.RawMatrix().Data[0]))

		if m, ok := s.momentums[p]; ok {
			// Multiply stored velocity by momentum (i.e., v*0.9)
			m.Scale(s.momentum, m)

			// Multiply gradient by 1-momentum (i.e., g*0.1)
			delta := mat.NewDense(deltaWs.RawMatrix().Rows, deltaWs.RawMatrix().Cols, nil)
			delta.Scale(1-s.momentum, deltaWs)

			// Calculate new gradient (v+g)
			m.Add(m, delta)

			// Update current gradient value with the value just computed
			deltaWs.Copy(m)

			// Save velocity
			s.momentums[p] = m
		} else {
			s.momentums[p] = mat.NewDense(deltaWs.RawMatrix().Rows, deltaWs.RawMatrix().Cols, slices.Clone(weights.RawMatrix().Data))
		}
	}

	tmp := mat.NewDense(deltaWs.RawMatrix().Rows, deltaWs.RawMatrix().Cols, nil)
	tmp.Scale(lr, deltaWs)
	weights.Sub(weights, tmp)
}
