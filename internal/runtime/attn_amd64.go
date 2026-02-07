//go:build amd64

package runtime

func init() {
	causalAttentionMultiHeadIntoImpl = causalAttentionMultiHeadIntoOptimized
}
