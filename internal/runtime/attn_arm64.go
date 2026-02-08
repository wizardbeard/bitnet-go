//go:build arm64

package runtime

func init() {
	if debugParityStrict {
		return
	}
	causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoOptimized
	storeCacheVectorImpl = storeCacheVectorOpt
	storeCacheVectorVImpl = storeCacheVectorVOpt
	softmaxInPlaceImpl = softmaxInPlaceOpt
}
