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
	nesterov  bool
}

func NewSGD(momentum float64, nesterov bool) Optimizer {
	return &SGD{
		momentum:  momentum,
		momentums: make(map[uintptr]*mat.Dense),
		nesterov:  nesterov,
	}
}

func (s *SGD) Apply(weights, deltaWs *mat.Dense, lr float64) {
	rows := deltaWs.RawMatrix().Rows
	cols := deltaWs.RawMatrix().Cols

	if s.momentum > 0 && s.momentum < 1 {
		p := uintptr(unsafe.Pointer(&weights.RawMatrix().Data[0]))

		if m, ok := s.momentums[p]; ok {
			// Multiply stored velocity by momentum (i.e., v*0.9)
			m.Scale(s.momentum, m)

			// Multiply gradient by 1-momentum (i.e., g*0.1)
			delta := mat.NewDense(rows, cols, nil)
			delta.Scale(1-s.momentum, deltaWs)

			// Calculate new gradient (v+g)
			m.Add(m, delta)

			// Update current gradient value with the value just computed
			if s.nesterov {
				// With Nesterov optimization we almost double the gradient, as its
				// formula is gradient + momentum * velocity
				nm := mat.NewDense(rows, cols, nil)
				nm.Scale(s.momentum, m)
				deltaWs.Add(deltaWs, nm)
			} else {
				deltaWs.Copy(m)
			}
		} else {
			s.momentums[p] = mat.NewDense(rows, cols, slices.Clone(weights.RawMatrix().Data))
		}
	}

	tmp := mat.NewDense(rows, cols, nil)
	tmp.Scale(lr, deltaWs)
	weights.Sub(weights, tmp)
}
