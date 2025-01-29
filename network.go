package deeper

import (
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
	Sizes     []int `json:"sizes"`
	Layers    []BackpropagationLayer
	optimizer Optimizer
	loss      Loss
	saver     Exporter
	callbacks []Callback

	// TODO Start to use pools and arenas to reduce memory allocations
	//matrices map[int]map[int]*sync.Pool
}

func NewNetwork() *Network {
	n := &Network{}

	return n
}

func (n *Network) AddLayer(l BackpropagationLayer) {
	if len(n.Layers) > 0 {
		parent := n.Layers[len(n.Layers)-1]
		parent.SetOutput(l)
		l.SetInput(parent)

		if !l.IsInput() && l.Weights() == nil {
			wi := NewNormWeightInitializer()
			l.SetWeights(wi.InitWeights(l.Rows(), parent.Rows()))
			l.SetBiases(wi.InitWeights(l.Rows(), 1))
		}
	}

	n.AddLayerWoWeightInitialization(l)
}

func (n *Network) AddLayerWoWeightInitialization(l BackpropagationLayer) {
	if len(n.Layers) > 0 {
		parent := n.Layers[len(n.Layers)-1]
		parent.SetOutput(l)
		l.SetInput(parent)
	}

	n.Layers = append(n.Layers, l)
}

func (n *Network) SetOptimizer(o Optimizer) {
	n.optimizer = o
}

func (n *Network) SetLossFunction(l Loss) {
	n.loss = l
}

type FitOptions struct {
	TrainX, TrainY []*mat.Dense
	ValX, ValY     []*mat.Dense
	Epochs         int
	BatchSize      int
	LearningRate   LearningRate
}

func (n *Network) Fit(o FitOptions) Evaluation {
	if !gt(o.TrainX, 0) || !gt(o.TrainY, 0) || !gt(o.ValX, 0) || !gt(o.ValY, 0) {
		log.Fatalln(fmt.Errorf("trainX, trainY, valX, and valY must be greater than zero"))
	}

	if len(o.TrainX) != len(o.TrainY) {
		log.Fatalln(fmt.Errorf("trainX and trainX must be of the same size"))
	}

	if len(o.ValX) != len(o.ValY) {
		log.Fatalln(fmt.Errorf("valX and valX must be of the same size"))
	}

	if !gt(o.Epochs, 0) {
		log.Fatalln(fmt.Errorf("epochs must be greater than zero"))
	}

	if !gt(o.BatchSize, 0) {
		log.Fatalln(fmt.Errorf("batch size must be greater than zero"))
	}

	var evaluation Evaluation
	var now time.Time
	var elapsed int64
	var batchSize int

	datasetSize := len(o.TrainX)

	for epoch := 1; epoch <= o.Epochs; epoch++ {
		rand.Shuffle(len(o.TrainX), func(i, j int) {
			o.TrainX[i], o.TrainX[j] = o.TrainX[j], o.TrainX[i]
			o.TrainY[i], o.TrainY[j] = o.TrainY[j], o.TrainY[i]
		})

		lr := o.LearningRate.LearningRate(o.Epochs, epoch)

		elapsed = 0
		now = time.Now()
		for i := 0; i < datasetSize; i += o.BatchSize {
			batchSize = o.BatchSize

			if i+o.BatchSize > datasetSize {
				batchSize = datasetSize - i
			}

			n.batch(o.TrainX[i:i+batchSize], o.TrainY[i:i+batchSize], lr)
		}
		elapsed = time.Since(now).Milliseconds()

		evaluation = n.Evaluate(o.ValX, o.ValY)

		fmt.Printf("Epoch %d (%.2f sec), val_acc: %.4f, lr: %.4f\n", epoch, float64(elapsed)/1000, evaluation.Accuracy, lr)

		for _, c := range n.callbacks {
			proceed := c.AfterEpoch(n, epoch, evaluation)

			if !proceed {
				goto END
			}
		}
	}

END:

	return evaluation
}

type backpropagationTask struct {
	x *mat.Dense
	y *mat.Dense
}

type backpropagationResult struct {
	deltaWs *Stack
	deltaBs *Stack
}

// batch computes and applies weight and bias updates over a single batch.
func (n *Network) batch(trainX []*mat.Dense, trainY []*mat.Dense, lr float64) {
	batchDeltaWs := make([]*mat.Dense, len(n.Layers)-1)
	batchDeltaBs := make([]*mat.Dense, len(n.Layers)-1)

	// There are no weights and no biases on the input layer
	for i, l := range n.Layers[1:] {
		batchDeltaWs[i] = mat.NewDense(l.Rows(), l.Cols(), nil) // [[30, 784], [10, 30]]
		batchDeltaBs[i] = mat.NewDense(l.Rows(), 1, nil)        // [[30, 1], [10, 1]]
	}

	taskWg := &sync.WaitGroup{}
	taskWg.Add(runtime.NumCPU())

	resultsWg := &sync.WaitGroup{}
	resultsWg.Add(1)

	taskCh := make(chan backpropagationTask, runtime.NumCPU())
	resultCh := make(chan backpropagationResult, runtime.NumCPU())

	// Create N workers for gradient computing, where N is the number of logical
	// cores (real ones + hyper threading)
	for range runtime.NumCPU() {
		go func() {
			defer taskWg.Done()

			for task := range taskCh {
				taskDeltaWs, taskDeltaBs := n.computeDeltas(task.x, task.y)
				resultCh <- backpropagationResult{taskDeltaWs, taskDeltaBs}
			}
		}()
	}

	// A goroutine to accumulate deltas within a single batch. This operation
	// is trivial comparing with gradient computing, so we likely don't need to
	// run it in parallel
	go func() {
		defer resultsWg.Done()

		for result := range resultCh {
			for i := range len(n.Layers) - 1 {
				batchDeltaWs[i].Add(batchDeltaWs[i], result.deltaWs.Pop())
				batchDeltaBs[i].Add(batchDeltaBs[i], result.deltaBs.Pop())
			}
		}
	}()

	for i := range len(trainX) {
		taskCh <- backpropagationTask{trainX[i], trainY[i]}
	}

	close(taskCh)
	taskWg.Wait()

	close(resultCh)
	resultsWg.Wait()

	for i := range len(n.Layers) - 1 {
		n.optimizer.Apply(n.Layers[i+1].Weights(), batchDeltaWs[i], lr/float64(len(trainX)))
		n.optimizer.Apply(n.Layers[i+1].Biases(), batchDeltaBs[i], lr/float64(len(trainX)))
	}
}

func (n *Network) computeDeltas(trainX *mat.Dense, trainY *mat.Dense) (*Stack, *Stack) {
	activations := NewStack(len(n.Layers))
	deltaWs := NewStack(len(n.Layers) - 1)
	deltaBs := NewStack(len(n.Layers) - 1)

	n.Layers[0].Feedforward(trainX, activations)
	diff := n.loss.Derivative(activations.Peek(), trainY)
	n.Layers[len(n.Layers)-1].Backpropagation(diff, activations, deltaWs, deltaBs)

	return deltaWs, deltaBs
}

type Evaluation struct {
	Accuracy  float32
	Matrix    map[int]map[int]int
	Recall    map[int]Counter
	Precision map[int]Counter
}

func (e Evaluation) ConfusionMatrix() string {
	buf := strings.Builder{}
	buf.WriteString("          ")

	for i := range 10 {
		buf.WriteString(fmt.Sprintf("%5d", i) + " ")
	}

	buf.WriteString(" Recall\n")

	for i := range 10 {
		for j := range 10 {
			if j == 0 {
				buf.WriteString(fmt.Sprintf("%9d", i) + " ")
			}

			if val, ok := e.Matrix[i][j]; ok {
				buf.WriteString(fmt.Sprintf("%5d", val) + " ")
			} else {
				buf.WriteString(fmt.Sprintf("%5d", 0) + " ")
			}
		}

		if val, ok := e.Recall[i]; ok {
			buf.WriteString(fmt.Sprintf("  %.2f", val.percent) + " ")
		}

		buf.WriteString("\n")
	}

	buf.WriteString("Precision ")

	for i := range 10 {
		if val, ok := e.Precision[i]; ok {
			buf.WriteString(fmt.Sprintf("%.2f", val.percent) + " ")
		} else {
			buf.WriteString(fmt.Sprintf("%5d", 0) + " ")
		}
	}

	return buf.String()
}

type Counter struct {
	correct int
	total   int
	percent float32
}

type evaluationTask struct {
	x *mat.Dense
	y *mat.Dense
}

type evaluationResult struct {
	pred  *mat.Dense
	truth *mat.Dense
}

func (n *Network) Evaluate(valX []*mat.Dense, valY []*mat.Dense) Evaluation {
	//var res *mat.Dense
	var evaluation Evaluation
	var correct int

	evaluation.Matrix = make(map[int]map[int]int, 10)
	evaluation.Recall = make(map[int]Counter, 10)
	evaluation.Precision = make(map[int]Counter, 10)

	for i := range 10 {
		evaluation.Matrix[i] = make(map[int]int, 10)
		evaluation.Recall[i] = Counter{}
		evaluation.Precision[i] = Counter{}
	}

	taskWg := &sync.WaitGroup{}
	taskWg.Add(runtime.NumCPU())

	resultsWg := &sync.WaitGroup{}
	resultsWg.Add(1)

	taskCh := make(chan evaluationTask, runtime.NumCPU())
	resultCh := make(chan evaluationResult, runtime.NumCPU())

	for range runtime.NumCPU() {
		go func() {
			defer taskWg.Done()

			for task := range taskCh {
				p := n.Layers[0].Feedforward(task.x, nil)
				resultCh <- evaluationResult{p, task.y}
			}
		}()
	}

	go func() {
		defer resultsWg.Done()

		for result := range resultCh {
			truth := Argmax(result.truth)
			prediction := Argmax(result.pred)

			evaluation.Matrix[truth][prediction] += 1

			if prediction == truth {
				correct++

				// Recall (per label)
				if v, ok := evaluation.Recall[truth]; ok {
					v.correct += 1
					evaluation.Recall[truth] = v
				}

				// Precision (per label)
				if v, ok := evaluation.Precision[prediction]; ok {
					v.correct += 1
					evaluation.Precision[prediction] = v
				}
			}

			// Recall (per label)
			if v, ok := evaluation.Recall[truth]; ok {
				v.total += 1
				evaluation.Recall[truth] = v
			}

			// Precision (per label)
			if v, ok := evaluation.Precision[prediction]; ok {
				v.total += 1
				evaluation.Precision[prediction] = v
			}
		}
	}()

	for i := range len(valX) {
		taskCh <- evaluationTask{valX[i], valY[i]}
	}

	close(taskCh)
	taskWg.Wait()

	close(resultCh)
	resultsWg.Wait()

	// Overall Accuracy
	evaluation.Accuracy = float32(correct) / float32(len(valX))

	for i := range 10 {
		// Recall (per label)
		if v, ok := evaluation.Recall[i]; ok {
			v.percent = (float32(v.correct) / float32(v.total)) * 100
			evaluation.Recall[i] = v
		}

		// Precision (per label)
		if v, ok := evaluation.Precision[i]; ok {
			v.percent = float32(v.correct) / float32(v.total) * 100
			evaluation.Precision[i] = v
		}
	}

	return evaluation
}

func (n *Network) AddCallback(c Callback) {
	n.callbacks = append(n.callbacks, c)
}
