//go:build arm64

package runtime

import (
	"strconv"
	"testing"
)

func BenchmarkCausalAttentionMultiHeadIntoCompareArm64(b *testing.B) {
	type cfg struct {
		steps int
		heads int
		dim   int
	}
	cases := []cfg{
		{steps: 64, heads: 8, dim: 64},
		{steps: 128, heads: 8, dim: 64},
		{steps: 256, heads: 16, dim: 64},
	}
	for _, c := range cases {
		name := "steps=" + strconv.Itoa(c.steps) + "/h=" + strconv.Itoa(c.heads) + "/d=" + strconv.Itoa(c.dim)
		b.Run(name, func(b *testing.B) {
			q := make([]float32, c.heads*c.dim)
			for i := range q {
				q[i] = float32(i%31) * 0.01
			}
			kStepDim := c.heads * c.dim
			vStepDim := c.heads * c.dim
			keys := make([]float32, c.steps*kStepDim)
			values := make([]float32, c.steps*vStepDim)
			for i := range keys {
				keys[i] = float32(i%37) * 0.01
				values[i] = float32(i%41) * 0.01
			}
			dst := make([]float32, len(q))
			scores := make([]float32, c.steps*c.heads)

			b.Run("generic", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(q) + len(keys) + len(values) + len(scores) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					causalAttentionMultiHeadIntoGeneric(dst, scores, q, keys, values, c.steps, c.heads, c.heads, kStepDim, vStepDim, 0)
				}
			})

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(q) + len(keys) + len(values) + len(scores) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					causalAttentionMultiHeadInto(dst, scores, q, keys, values, c.steps, c.heads, c.heads, kStepDim, vStepDim, 0)
				}
			})
		})
	}
}
