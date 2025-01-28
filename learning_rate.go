package deeper

import "math"

type LearningRate interface {
	LearningRate(epochs, epoch int) float64
}

type FlatLearningRate struct {
	lr float64
}

func NewFlatLearningRate(rate float64) LearningRate {
	return &FlatLearningRate{rate}
}

func (f *FlatLearningRate) LearningRate(_, _ int) float64 {
	return f.lr
}

type CosineDecayLearningRate struct {
	initial float64
	final   float64
}

func NewCosineDecayLearningRate(initial, final float64) LearningRate {
	return &CosineDecayLearningRate{
		initial: initial,
		final:   final,
	}
}

// LearningRate gradually decays initial learning rate to its final value through
// calculations of intermediate steps using the cosine decay schedule proposed by
// Loshchilov et al. (https://doi.org/10.48550/arXiv.1608.03983) but without warm
// restarts
func (c *CosineDecayLearningRate) LearningRate(epochs, epoch int) float64 {
	if epoch == 1 {
		return c.initial
	}

	return c.final +
		0.5*(c.initial-c.final)*
			(1+math.Cos(math.Pi*float64(epoch)/float64(epochs)))
}
