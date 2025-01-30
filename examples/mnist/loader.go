package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io"
	"os"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

type RawImage []byte
type Label uint8

func Load(set string) ([]*mat.Dense, []*mat.Dense, error) {
	var imageData []RawImage
	var labelData []Label
	var err error

	r, err := os.Open(fmt.Sprintf("%s-images-idx3-ubyte.gz", set))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open image set %s: %w", set, err)
	}
	gzr, err := gzip.NewReader(r)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create gzip reader for set %s: %w", set, err)
	}

	if imageData, err = readImageFile(gzr); err != nil {
		return nil, nil, fmt.Errorf("failed to read images for set %s: %w", set, err)
	}

	r, err = os.Open(fmt.Sprintf("%s-labels-idx1-ubyte.gz", set))
	if err != nil {
		return nil, nil, fmt.Errorf("")
	}
	gzr, _ = gzip.NewReader(r)

	if labelData, err = readLabelFile(gzr); err != nil {
		return nil, nil, err
	}

	x := prepareX(imageData)
	y := prepareY(labelData)

	return x, y, nil
}

func readImageFile(r io.Reader) (imgs []RawImage, err error) {
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != m {
			return nil, os.ErrInvalid
		}
	}
	return imgs, nil
}

func readLabelFile(r io.Reader) (labels []Label, err error) {
	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err = binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

func prepareX(xx []RawImage) []*mat.Dense {
	rows := len(xx)
	length := len(xx[0])
	data := make([]float64, rows*length)
	images := make([]*mat.Dense, rows)

	for i := range rows {
		images[i] = mat.NewDense(length, 1, data[i*length:i*length+length:i*length+length])

		for j := range length {
			if xx[i][j] > 0 {
				images[i].Set(j, 0, float64(xx[i][j])/255)
			}
		}
	}

	return images
}

func prepareY(yy []Label) []*mat.Dense {
	rows := len(yy)
	labels := make([]*mat.Dense, rows)

	for i := range rows {
		labels[i] = mat.NewDense(10, 1, nil)
		labels[i].Set(int(yy[i]), 0, 1)
	}

	return labels
}
