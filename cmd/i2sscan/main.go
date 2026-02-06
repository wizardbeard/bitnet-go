package main

import (
	"fmt"
	"os"

	"bitnet-go/internal/gguf"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: i2sscan <model> <tensor>")
		os.Exit(2)
	}
	modelPath := os.Args[1]
	name := os.Args[2]

	info, err := gguf.ReadModelInfo(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read model info: %v\n", err)
		os.Exit(1)
	}
	packed, _, count, err := gguf.ReadTensorI2SPacked(modelPath, info, name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read i2_s tensor: %v\n", err)
		os.Exit(1)
	}

	var counts [4]uint64
	for i := uint64(0); i < count; i++ {
		const block = 128
		const blockBytes = 32
		bi := int(i) / block
		off := int(i) % block
		gp := off % 32
		group := off / 32
		p := bi*blockBytes + gp
		q := (packed[p] >> uint(6-2*group)) & 0x3
		counts[q]++
	}
	fmt.Printf("counts: 0=%d 1=%d 2=%d 3=%d\n", counts[0], counts[1], counts[2], counts[3])
}
