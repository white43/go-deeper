package deeper

import (
	"encoding/json"
	"fmt"
	"io"

	"gonum.org/v1/gonum/mat"
)

type Exporter interface {
	Save(dst io.Writer, src *Network) error
	Load(dst *Network, src io.Reader) error
}

type Export struct{}

type jsonMatrix struct {
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
	Data []float64 `json:"data"`
}

type jsonNetwork struct {
	Sizes   []int        `json:"sizes"`
	Weights []jsonMatrix `json:"weights"`
	Biases  []jsonMatrix `json:"biases"`
}

// NewExporter returns an interface for saving and loading trained models
func NewExporter() Exporter {
	return &Export{}
}

// Save exports an existing and likely trained network to a destination
// (i.e., a file). This is responsibility of the caller to close the writer
func (e *Export) Save(dst io.Writer, src *Network) error {
	j := jsonNetwork{}

	for _, l := range src.Layers {
		j.Sizes = append(j.Sizes, l.Rows())

		if !l.IsInput() {
			j.Weights = append(j.Weights, jsonMatrix{Data: l.Weights().RawMatrix().Data})
			j.Biases = append(j.Biases, jsonMatrix{Data: l.Biases().RawMatrix().Data})
		}
	}

	if err := json.NewEncoder(dst).Encode(j); err != nil {
		return fmt.Errorf("could not marshal network: %w", err)
	}

	return nil
}

// Load loads a previously exported network from its saved state for inference.
// This is responsibility of the caller to close the reader.
func (e *Export) Load(dst *Network, src io.Reader) error {
	j := jsonNetwork{}

	if err := json.NewDecoder(src).Decode(&j); err != nil {
		return fmt.Errorf("couldn't decode saved network: %w", err)
	}

	dst.Layers = make([]BackpropagationLayer, 0)

	for i := range j.Sizes {
		var l BackpropagationLayer

		if i == 0 {
			l = NewInputLayer(j.Sizes[i])
		} else if i == len(j.Sizes)-1 {
			l = NewOutputLayer(j.Sizes[i], NewSoftmax())
		} else {
			l = NewHiddenLayer(j.Sizes[i], NewSigmoid())
		}

		if !l.IsInput() {
			l.SetWeights(mat.NewDense(j.Sizes[i], j.Sizes[i-1], j.Weights[i-1].Data))
			l.SetBiases(mat.NewDense(j.Sizes[i], 1, j.Biases[i-1].Data))
		}

		dst.AddLayerWoWeightInitialization(l)
	}

	return nil
}
