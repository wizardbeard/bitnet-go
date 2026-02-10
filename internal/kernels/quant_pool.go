package kernels

import (
	"os"
	"runtime"
	"sync"
)

type i2sI8STask struct {
	transposed bool
	dst        []float32
	packed     []byte
	rows       int
	cols       int
	vec        []int8
	weight     float32
	act        float32
	actSum     int32
	start      int
	end        int
	wg         *sync.WaitGroup
}

var i2sI8SPoolEnabled = os.Getenv("BITNET_I2S_I8S_POOL") != "0"
var i2sI8SPoolWorkers = envInt("BITNET_I2S_I8S_POOL_WORKERS", 0)

var (
	i2sI8SPoolOnce sync.Once
	i2sI8SPoolCh   chan i2sI8STask
)

func initI2SI8SPool() {
	workers := i2sI8SPoolWorkers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers < 1 {
		workers = 1
	}
	i2sI8SPoolCh = make(chan i2sI8STask, workers*2)
	for i := 0; i < workers; i++ {
		go func() {
			for task := range i2sI8SPoolCh {
				runI2SI8STask(task)
			}
		}()
	}
}

func runI2SI8STask(task i2sI8STask) {
	if task.transposed {
		matVecTI2SI8SRange(task.dst, task.packed, task.rows, task.cols, task.vec, task.weight, task.act, task.actSum, task.start, task.end)
	} else {
		matVecI2SI8SRange(task.dst, task.packed, task.rows, task.cols, task.vec, task.weight, task.act, task.actSum, task.start, task.end)
	}
	task.wg.Done()
}

func submitI2SI8STask(task i2sI8STask) {
	if !i2sI8SPoolEnabled {
		runI2SI8STask(task)
		return
	}
	i2sI8SPoolOnce.Do(initI2SI8SPool)
	i2sI8SPoolCh <- task
}
