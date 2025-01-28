package main

import (
	"fmt"
	"log"
	"os"
)

type Callback interface {
	AfterEpoch(n *Network, epoch int, ev Evaluation) bool
}

type SaveBest struct {
	bestAccuracy float32
	threshold    float32
	saver        Exporter
}

// NewSaveBest exports models through the Exporter interface once their accuracy
// exceeds the defined threshold value. SaveBest tracks the best accuracy during
// training sessions and exports models only when their accuracy is higher than
// previous best value in that session.
func NewSaveBest(saver Exporter, threshold float32) Callback {
	return &SaveBest{
		threshold: threshold,
		saver:     saver,
	}
}

func (sb *SaveBest) AfterEpoch(n *Network, _ int, ev Evaluation) bool {
	if ev.accuracy > sb.threshold && ev.accuracy > sb.bestAccuracy {
		sb.bestAccuracy = ev.accuracy
		sb.doSaveBest(n, ev)
	}

	return true
}

func (sb *SaveBest) doSaveBest(n *Network, e Evaluation) {
	fp, err := os.OpenFile(sb.getFilename(e), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		log.Fatalln(fmt.Errorf("failed to open file %s: %w", sb.getFilename(e), err))
	}
	defer fp.Close()

	if err = sb.saver.Save(fp, n); err != nil {
		log.Fatalln(fmt.Errorf("failed to save network: %w", err))
	}
}

func (sb *SaveBest) getFilename(e Evaluation) string {
	return fmt.Sprintf("model-%.4f.json", e.accuracy)
}

type EarlyStopping struct {
	bestAccuracy float32
	bestEpoch    int
	waitEpochs   int
}

// NewEarlyStopping creates a Callback that may interrupt training process
// when we have not seen any improvements in accuracy for waitEpochs epochs
func NewEarlyStopping(waitEpochs int) Callback {
	return &EarlyStopping{waitEpochs: waitEpochs}
}

func (es *EarlyStopping) AfterEpoch(n *Network, epoch int, ev Evaluation) bool {
	if epoch-es.bestEpoch >= es.waitEpochs {
		return false
	}

	if ev.accuracy > es.bestAccuracy {
		es.bestAccuracy = ev.accuracy
		es.bestEpoch = epoch
	}

	return true
}
