package main

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type WeightInitializer interface {
	InitWeights(r, c int) *mat.Dense
}

type NormWeightInitializer struct{}

func NewNormWeightInitializer() WeightInitializer {
	return &NormWeightInitializer{}
}

func (n *NormWeightInitializer) InitWeights(r, c int) *mat.Dense {
	data := make([]float64, r*c)

	for i := range r * c {
		data[i] = rand.NormFloat64()
	}

	return mat.NewDense(r, c, data)
}
