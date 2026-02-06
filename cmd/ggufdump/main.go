package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"bitnet-go/internal/gguf"
)

func main() {
	var (
		modelPath = flag.String("model", "", "Path to GGUF model")
		showKV    = flag.Bool("kv", true, "Print GGUF key-values")
		kvPrefix  = flag.String("kv-prefix", "", "Only print KV keys with this prefix")
		showTensors = flag.Bool("tensors", true, "Print tensor directory")
	)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "missing required --model")
		flag.Usage()
		os.Exit(2)
	}

	info, err := gguf.ReadModelInfo(*modelPath)
	if err != nil {
		log.Fatalf("read gguf model info: %v", err)
	}

	fmt.Printf("model=%s version=%d tensors=%d kv=%d\n", *modelPath, info.Version, info.TensorCount, info.KVCount)

	if *showKV {
		fmt.Println("kv:")
		keys := make([]string, 0, len(info.KeyValues))
		for k := range info.KeyValues {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if *kvPrefix != "" && !hasPrefix(k, *kvPrefix) {
				continue
			}
			fmt.Printf("  %s = %v\n", k, info.KeyValues[k])
		}
	}

	if *showTensors {
		fmt.Println("tensors:")
		tensors := append([]gguf.TensorInfo(nil), info.Tensors...)
		sort.Slice(tensors, func(i, j int) bool {
			return tensors[i].Name < tensors[j].Name
		})
		for _, t := range tensors {
			fmt.Printf("  %s dims=%v type=%d offset=%d\n", t.Name, t.Dimensions, t.Type, t.Offset)
		}
	}
}

func hasPrefix(s, prefix string) bool {
	if len(prefix) == 0 {
		return true
	}
	if len(s) < len(prefix) {
		return false
	}
	return s[:len(prefix)] == prefix
}
