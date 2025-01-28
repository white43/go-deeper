package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

func gt(a any, min int) bool {
	if a == nil {
		return false
	}

	switch t := a.(type) {
	case int:
		return t > min
	case string:
		return len(t) > min
	case []*mat.Dense:
		return len(t) > min
	default:
		log.Fatal(fmt.Errorf("gt: unknown type: %T", t))
	}

	return false
}

func Argmax(m *mat.Dense) int {
	col := m.ColView(0)
	var maxIdx int
	var lastMax float64

	for i := range col.Len() {
		if curMax := col.AtVec(i); curMax > lastMax {
			lastMax = curMax
			maxIdx = i
		}
	}

	return maxIdx
}
