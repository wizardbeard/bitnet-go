//go:build arm64

package runtime

func init() {
	causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoOptimized
}
