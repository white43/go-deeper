package deeper

import "gonum.org/v1/gonum/mat"

type Stack struct {
	next  int
	size  int
	stack []*mat.Dense
}

func NewStack(size int) *Stack {
	return &Stack{
		size:  size,
		stack: make([]*mat.Dense, size),
	}
}

func (a *Stack) Push(m *mat.Dense) {
	if a.next+1 > a.size {
		panic("stack overflow")
	}

	a.stack[a.next] = m
	a.next++
}

func (a *Stack) Pop() *mat.Dense {
	if a.next-1 < 0 {
		panic("stack underflow")
	}

	t := a.stack[a.next-1]
	a.next--
	a.stack = a.stack[:a.next]
	return t
}

func (a *Stack) Peek() *mat.Dense {
	return a.stack[a.next-1]
}
