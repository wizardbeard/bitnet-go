#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llama.h"

namespace {

void silent_log_callback(enum ggml_log_level, const char *, void *) {}

std::string env_or(const char * key, const char * fallback) {
    const char * v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return v;
}

} // namespace

int main() {
    const std::string model_path = env_or("BITNET_REF_MODEL", "");
    const std::string prompt = env_or("BITNET_REF_PROMPT", "");

    if (model_path.empty()) {
        std::fprintf(stderr, "BITNET_REF_MODEL is required\n");
        return 1;
    }

    llama_log_set(silent_log_callback, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const int cap = 65536;
    std::vector<llama_token> tokens(static_cast<size_t>(cap));
    int n = llama_tokenize(model, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), cap, true, false);
    if (n < 0) {
        n = -n;
        tokens.resize(static_cast<size_t>(n));
        n = llama_tokenize(model, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), n, true, false);
    }
    if (n < 0) {
        std::fprintf(stderr, "tokenize failed\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        std::printf("PROMPT_TOKEN idx=%d id=%d\n", i, static_cast<int>(tokens[static_cast<size_t>(i)]));
    }

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
