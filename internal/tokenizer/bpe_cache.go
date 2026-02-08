package tokenizer

type bpeChunkCache struct {
	capacity int
	order    []string
	index    int
	values   map[string][]int32
}

func newBPEChunkCache(capacity int) *bpeChunkCache {
	if capacity <= 0 {
		capacity = 128
	}
	return &bpeChunkCache{
		capacity: capacity,
		order:    make([]string, capacity),
		values:   make(map[string][]int32, capacity),
	}
}

func (c *bpeChunkCache) get(key string) []int32 {
	if c == nil {
		return nil
	}
	if v, ok := c.values[key]; ok {
		return v
	}
	return nil
}

func (c *bpeChunkCache) add(key string, val []int32) {
	if c == nil {
		return
	}
	if _, ok := c.values[key]; ok {
		return
	}
	if c.capacity <= 0 {
		return
	}
	slot := c.index % c.capacity
	if old := c.order[slot]; old != "" {
		delete(c.values, old)
	}
	c.order[slot] = key
	c.values[key] = cloneInt32(val)
	c.index++
}

func cloneInt32(src []int32) []int32 {
	if len(src) == 0 {
		return nil
	}
	dst := make([]int32, len(src))
	copy(dst, src)
	return dst
}
