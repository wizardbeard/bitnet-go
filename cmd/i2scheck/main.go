package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"bitnet-go/internal/gguf"
	"bitnet-go/internal/kernels"
)

func main() {
	var (
		modelPath = flag.String("model", "", "Path to GGUF model")
		tensor    = flag.String("tensor", "", "Tensor name (i2_s)")
		rowStart  = flag.Int("row", 0, "Row index to inspect")
		rows      = flag.Int("rows", 1, "Number of rows to report")
		seed      = flag.Int64("seed", 1, "Random seed for input vector")
	)
	flag.Parse()

	if *modelPath == "" || *tensor == "" {
		fmt.Fprintln(os.Stderr, "usage: i2scheck --model <path> --tensor <name> [--row N] [--rows N] [--seed N]")
		flag.Usage()
		os.Exit(2)
	}

	info, err := gguf.ReadModelInfo(*modelPath)
	if err != nil {
		log.Fatalf("read model info: %v", err)
	}
	ti, ok := info.TensorByName(*tensor)
	if !ok {
		log.Fatalf("tensor not found: %s", *tensor)
	}
	if ti.Type != gguf.GGMLTypeI2_S {
		log.Fatalf("tensor %s type=%d is not i2_s", *tensor, ti.Type)
	}
	if len(ti.Dimensions) != 2 {
		log.Fatalf("tensor %s expected 2 dims, got %v", *tensor, ti.Dimensions)
	}

	packed, scale, _, err := gguf.ReadTensorI2SPacked(*modelPath, info, *tensor)
	if err != nil {
		log.Fatalf("read i2_s packed: %v", err)
	}
	mat, err := gguf.ReadTensorAsF32(*modelPath, info, *tensor)
	if err != nil {
		log.Fatalf("read i2_s as f32: %v", err)
	}

	rowsN := int(ti.Dimensions[0])
	colsN := int(ti.Dimensions[1])
	if rowsN <= 0 || colsN <= 0 {
		log.Fatalf("invalid dims %v", ti.Dimensions)
	}
	if len(mat) < rowsN*colsN {
		log.Fatalf("unexpected f32 length %d for dims %dx%d", len(mat), rowsN, colsN)
	}

	if *rowStart < 0 || *rowStart >= rowsN {
		log.Fatalf("row out of range: %d (rows=%d)", *rowStart, rowsN)
	}
	if *rows < 1 {
		*rows = 1
	}
	if *rowStart+*rows > rowsN {
		*rows = rowsN - *rowStart
	}

	rng := rand.New(rand.NewSource(*seed))
	vec := make([]float32, colsN)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}

	dstPacked := make([]float32, rowsN)
	kernels.MatVecI2S(dstPacked, packed, rowsN, colsN, vec, scale)

	fmt.Printf("tensor=%s dims=%dx%d seed=%d time=%s\n", *tensor, rowsN, colsN, *seed, time.Now().Format(time.RFC3339))
	fmt.Printf("rows %d..%d:\n", *rowStart, *rowStart+*rows-1)
	for r := *rowStart; r < *rowStart+*rows; r++ {
		var sum float64
		base := r
		for c := 0; c < colsN; c++ {
			sum += float64(mat[base]) * float64(vec[c])
			base += rowsN
		}
		f32 := float32(sum)
		packedVal := dstPacked[r]
		delta := f32 - packedVal
		fmt.Printf("  row=%d f32=%g packed=%g delta=%g\n", r, f32, packedVal, delta)
	}
}
