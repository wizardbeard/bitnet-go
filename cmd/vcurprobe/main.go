package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"bitnet-go/internal/gguf"
)

func parseCSV(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return nil, fmt.Errorf("empty csv: %s", path)
	}
	parts := strings.Split(s, ",")
	out := make([]float32, 0, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 32)
		if err != nil {
			return nil, fmt.Errorf("%s idx=%d parse %q: %w", path, i, p, err)
		}
		out = append(out, float32(v))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no values parsed from: %s", path)
	}
	return out, nil
}

func vecAbsDiffStats(a, b []float32) (meanAbs, maxAbs float32, maxIdx int) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n <= 0 {
		return 0, 0, -1
	}
	var sum float32
	maxIdx = -1
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
		if d > maxAbs {
			maxAbs = d
			maxIdx = i
		}
	}
	return sum / float32(n), maxAbs, maxIdx
}

func main() {
	var modelPath string
	var layer int
	var attnNormCSV string
	var vcurRefCSV string
	flag.StringVar(&modelPath, "model", "", "path to GGUF model")
	flag.IntVar(&layer, "layer", 14, "layer index")
	flag.StringVar(&attnNormCSV, "attn-norm-csv", "", "path to attn_norm csv values")
	flag.StringVar(&vcurRefCSV, "vcur-ref-csv", "", "path to Vcur csv values")
	flag.Parse()

	if modelPath == "" || attnNormCSV == "" || vcurRefCSV == "" {
		fmt.Fprintln(os.Stderr, "usage: vcurprobe --model <gguf> --layer <n> --attn-norm-csv <file> --vcur-ref-csv <file>")
		os.Exit(2)
	}

	attnNorm, err := parseCSV(attnNormCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read attn_norm: %v\n", err)
		os.Exit(1)
	}
	vcurRef, err := parseCSV(vcurRefCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read vcur_ref: %v\n", err)
		os.Exit(1)
	}

	info, err := gguf.ReadModelInfo(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ReadModelInfo: %v\n", err)
		os.Exit(1)
	}
	f, err := os.Open(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open model: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	tensorName := fmt.Sprintf("blk.%d.attn_v.weight", layer)
	ti, ok := info.TensorByName(tensorName)
	if !ok {
		fmt.Fprintf(os.Stderr, "tensor not found: %s\n", tensorName)
		os.Exit(1)
	}
	if len(ti.Dimensions) != 2 {
		fmt.Fprintf(os.Stderr, "tensor dims unsupported: %v\n", ti.Dimensions)
		os.Exit(1)
	}
	rows := int(ti.Dimensions[0])
	cols := int(ti.Dimensions[1])
	w, err := gguf.ReadTensorAsF32FromFile(f, info, tensorName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read tensor as f32: %v\n", err)
		os.Exit(1)
	}
	if len(w) != rows*cols {
		fmt.Fprintf(os.Stderr, "unexpected weight len=%d rows=%d cols=%d\n", len(w), rows, cols)
		os.Exit(1)
	}

	inDim := len(attnNorm)
	transposed := false
	if rows == inDim {
		transposed = true
	} else if cols == inDim {
		transposed = false
	} else {
		fmt.Fprintf(os.Stderr, "incompatible dims rows=%d cols=%d inDim=%d\n", rows, cols, inDim)
		os.Exit(1)
	}

	var outDim int
	if transposed {
		outDim = cols
	} else {
		outDim = rows
	}
	vcur := make([]float32, outDim)
	if transposed {
		for o := 0; o < outDim; o++ {
			base := o * rows
			var sum float32
			for i := 0; i < rows; i++ {
				sum += w[base+i] * attnNorm[i]
			}
			vcur[o] = sum
		}
	} else {
		for r := 0; r < rows; r++ {
			var sum float32
			for c := 0; c < cols; c++ {
				sum += w[r+rows*c] * attnNorm[c]
			}
			vcur[r] = sum
		}
	}

	meanAbs, maxAbs, maxIdx := vecAbsDiffStats(vcur, vcurRef)
	fmt.Printf("vcurprobe layer=%d tensor=%s rows=%d cols=%d transposed=%v in_dim=%d out_dim=%d ref_n=%d mean_abs=%g max_abs=%g max_idx=%d\n",
		layer, tensorName, rows, cols, transposed, inDim, outDim, len(vcurRef), meanAbs, maxAbs, maxIdx)
	if maxIdx >= 0 && maxIdx < len(vcur) && maxIdx < len(vcurRef) {
		fmt.Printf("vcurprobe max_idx_values got=%g ref=%g\n", vcur[maxIdx], vcurRef[maxIdx])
	}
}
