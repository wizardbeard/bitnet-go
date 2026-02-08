package runtime

import "fmt"

var causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoGeneric

func causalAttentionMultiHeadInto(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int, pos int) {
	causalAttentionMultiHeadIntoImpl(dst, scores, q, keys, values, steps, qHeads, kvHeads, kStepDim, vStepDim, pos)
	if debugAttnRef && shouldDebug(pos) {
		ref := make([]float32, len(dst))
		causalAttentionMultiHeadIntoReference(ref, q, keys, values, steps, qHeads, kvHeads, kStepDim, vStepDim)
		debugVecDiff(fmt.Sprintf("attn_ref.diff.pos=%d", pos), dst, ref)
	}
}
