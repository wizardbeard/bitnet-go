//go:build amd64

package runtime

func init() {
	causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoOptimized
	storeCacheVectorImpl = storeCacheVectorOpt
	storeCacheVectorVImpl = storeCacheVectorVOpt
	softmaxInPlaceImpl = softmaxInPlaceOpt
}
