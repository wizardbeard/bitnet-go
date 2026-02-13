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

func projectTensor(
	f *os.File,
	info gguf.ModelInfo,
	tensorName string,
	input []float32,
) ([]float32, int, int, bool, error) {
	ti, ok := info.TensorByName(tensorName)
	if !ok {
		return nil, 0, 0, false, fmt.Errorf("tensor not found: %s", tensorName)
	}
	if len(ti.Dimensions) != 2 {
		return nil, 0, 0, false, fmt.Errorf("tensor dims unsupported for %s: %v", tensorName, ti.Dimensions)
	}
	rows := int(ti.Dimensions[0])
	cols := int(ti.Dimensions[1])
	w, err := gguf.ReadTensorAsF32FromFile(f, info, tensorName)
	if err != nil {
		return nil, 0, 0, false, fmt.Errorf("read tensor %s as f32: %w", tensorName, err)
	}
	if len(w) != rows*cols {
		return nil, 0, 0, false, fmt.Errorf("unexpected len for %s: len=%d rows=%d cols=%d", tensorName, len(w), rows, cols)
	}

	inDim := len(input)
	transposed := false
	if rows == inDim {
		transposed = true
	} else if cols == inDim {
		transposed = false
	} else {
		return nil, 0, 0, false, fmt.Errorf("incompatible dims for %s: rows=%d cols=%d in=%d", tensorName, rows, cols, inDim)
	}

	var outDim int
	if transposed {
		outDim = cols
	} else {
		outDim = rows
	}
	out := make([]float32, outDim)
	if transposed {
		for o := 0; o < outDim; o++ {
			base := o * rows
			var sum float32
			for i := 0; i < rows; i++ {
				sum += w[base+i] * input[i]
			}
			out[o] = sum
		}
	} else {
		for r := 0; r < rows; r++ {
			var sum float32
			for c := 0; c < cols; c++ {
				sum += w[r+rows*c] * input[c]
			}
			out[r] = sum
		}
	}
	return out, rows, cols, transposed, nil
}

func printProbe(label string, got, ref []float32) {
	meanAbs, maxAbs, maxIdx := vecAbsDiffStats(got, ref)
	fmt.Printf("qkvprobe label=%s got_n=%d ref_n=%d mean_abs=%g max_abs=%g max_idx=%d\n",
		label, len(got), len(ref), meanAbs, maxAbs, maxIdx)
	if maxIdx >= 0 && maxIdx < len(got) && maxIdx < len(ref) {
		fmt.Printf("qkvprobe label=%s max_idx_values got=%g ref=%g\n", label, got[maxIdx], ref[maxIdx])
	}
}

func main() {
	var modelPath string
	var layer int
	var inputCSV string
	var qRefCSV string
	var kRefCSV string
	var vRefCSV string
	var label string
	flag.StringVar(&modelPath, "model", "", "path to GGUF model")
	flag.IntVar(&layer, "layer", 14, "layer index")
	flag.StringVar(&inputCSV, "input-csv", "", "path to attn_norm csv values")
	flag.StringVar(&qRefCSV, "q-ref-csv", "", "path to Qcur csv values to compare")
	flag.StringVar(&kRefCSV, "k-ref-csv", "", "path to Kcur csv values to compare")
	flag.StringVar(&vRefCSV, "v-ref-csv", "", "path to Vcur csv values to compare")
	flag.StringVar(&label, "label", "", "free-form run label")
	flag.Parse()

	if modelPath == "" || inputCSV == "" || qRefCSV == "" || kRefCSV == "" || vRefCSV == "" {
		fmt.Fprintln(os.Stderr, "usage: qkvprobe --model <gguf> --layer <n> --input-csv <file> --q-ref-csv <file> --k-ref-csv <file> --v-ref-csv <file> [--label <tag>]")
		os.Exit(2)
	}

	input, err := parseCSV(inputCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read input: %v\n", err)
		os.Exit(1)
	}
	qRef, err := parseCSV(qRefCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read q_ref: %v\n", err)
		os.Exit(1)
	}
	kRef, err := parseCSV(kRefCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read k_ref: %v\n", err)
		os.Exit(1)
	}
	vRef, err := parseCSV(vRefCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read v_ref: %v\n", err)
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

	qName := fmt.Sprintf("blk.%d.attn_q.weight", layer)
	kName := fmt.Sprintf("blk.%d.attn_k.weight", layer)
	vName := fmt.Sprintf("blk.%d.attn_v.weight", layer)

	qCur, qRows, qCols, qT, err := projectTensor(f, info, qName, input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "project q: %v\n", err)
		os.Exit(1)
	}
	kCur, kRows, kCols, kT, err := projectTensor(f, info, kName, input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "project k: %v\n", err)
		os.Exit(1)
	}
	vCur, vRows, vCols, vT, err := projectTensor(f, info, vName, input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "project v: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("qkvprobe run label=%s layer=%d in_dim=%d q_rows=%d q_cols=%d q_t=%v k_rows=%d k_cols=%d k_t=%v v_rows=%d v_cols=%d v_t=%v\n",
		label, layer, len(input), qRows, qCols, qT, kRows, kCols, kT, vRows, vCols, vT)
	printProbe(label+".Qcur", qCur, qRef)
	printProbe(label+".Kcur", kCur, kRef)
	printProbe(label+".Vcur", vCur, vRef)
}
