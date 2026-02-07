package runtime

var causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoGeneric

func causalAttentionMultiHeadInto(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int, pos int) {
	causalAttentionMultiHeadIntoImpl(dst, scores, q, keys, values, steps, qHeads, kvHeads, kStepDim, vStepDim, pos)
}
