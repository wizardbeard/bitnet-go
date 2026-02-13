#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"
#include "llama.h"

namespace {

void silent_log_callback(enum ggml_log_level, const char *, void *) {
}

std::string env_or(const char * key, const char * fallback) {
    const char * v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return v;
}

int env_or_int(const char * key, int fallback) {
    const char * v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return std::atoi(v);
}

bool env_or_bool(const char * key, bool fallback) {
    const char * v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return std::atoi(v) != 0;
}

struct TopKEntry {
    int id;
    float logit;
};

std::vector<TopKEntry> topk_from_logits(const float * logits, int n_vocab, int k) {
    if (k <= 0) {
        k = 1;
    }
    if (k > n_vocab) {
        k = n_vocab;
    }

    std::vector<TopKEntry> best;
    best.reserve(static_cast<size_t>(k));

    for (int i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (static_cast<int>(best.size()) < k) {
            best.push_back({i, v});
            continue;
        }
        auto min_it = std::min_element(best.begin(), best.end(), [](const TopKEntry & a, const TopKEntry & b) {
            return a.logit < b.logit;
        });
        if (v > min_it->logit) {
            *min_it = {i, v};
        }
    }

    std::sort(best.begin(), best.end(), [](const TopKEntry & a, const TopKEntry & b) {
        return a.logit > b.logit;
    });

    return best;
}

void print_topk_line(int step, const std::vector<TopKEntry> & entries) {
    std::printf("TOPK step=%d entries=", step);
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("%d:%.9g", entries[i].id, entries[i].logit);
    }
    std::printf("\n");
}

struct DebugState {
    bool enabled = false;
    int target_pos = -1;
    int current_pos = -1;
    std::unordered_set<std::string> seen;
    bool i2s_dot = false;
    bool i2s_dot_done = false;
    std::string i2s_dot_tensor;
    int i2s_dot_row = 0;
    llama_model * model = nullptr;
};

bool g_print_values = false;
int g_values_n = 8;
std::string g_values_name;

bool starts_with(const char * s, const char * prefix) {
    if (s == nullptr || prefix == nullptr) {
        return false;
    }
    while (*prefix) {
        if (*s == '\0' || *s != *prefix) {
            return false;
        }
        ++s;
        ++prefix;
    }
    return true;
}

bool has_numeric_suffix(const char * s) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    while (*s) {
        if (!std::isdigit(static_cast<unsigned char>(*s))) {
            return false;
        }
        ++s;
    }
    return true;
}

bool matches_layered_tensor(const char * name, const char * base) {
    if (!starts_with(name, base)) {
        return false;
    }
    const size_t n = std::strlen(base);
    if (name[n] != '-') {
        return false;
    }
    return has_numeric_suffix(name + n + 1);
}

bool name_matches(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * targets[] = {
        "inp_embd",
        "attn_norm-0",
        "attn_sub_norm-0",
        "attn_o_out-0",
        "Qcur-0",
        "Kcur-0",
        "Vcur-0",
        "q-0",
        "k-0",
        "v-0",
        "kq_soft_max_ext-0",
        "kqv-0",
        "kqv_wo-0",
        "kqv_out-0",
        "attn_o_out-0",
        "ffn_inp-0",
        "ffn_norm-0",
        "ffn_up-0",
        "ffn_gate-0",
        "ffn_act-0",
        "ffn_down-0",
        "ffn_sub_norm-0",
        "ffn_out-0",
        "l_out-0",
        "result_norm",
        "result_output",
    };
    for (const char * target : targets) {
        if (std::strcmp(name, target) == 0) {
            return true;
        }
    }
    const char * layered[] = {
        "attn_norm",
        "attn_sub_norm",
        "kq_soft_max_ext",
        "attn_o_out",
        "Qcur",
        "Kcur",
        "Vcur",
        "ffn_inp",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_act",
        "ffn_down",
        "ffn_sub_norm",
        "ffn_out",
        "l_out",
    };
    for (const char * base : layered) {
        if (matches_layered_tensor(name, base)) {
            return true;
        }
    }
    return false;
}

bool name_values_matches(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    if (!g_values_name.empty()) {
        return g_values_name == name;
    }
    const char * targets[] = {
        "kq_soft_max_ext-0",
        "kqv-0",
        "attn_o_out-0",
        "inp_embd",
        "attn_norm-0",
        "ffn_inp-0",
        "ffn_gate-0",
        "ffn_up-0",
        "ffn_act-0",
        "ffn_down-0",
        "attn_sub_norm-0",
        "Qcur-0",
        "Kcur-0",
        "Vcur-0",
        "ffn_norm-0",
        "ffn_out-0",
        "ffn_sub_norm-0",
        "result_norm",
    };
    for (const char * target : targets) {
        if (std::strcmp(name, target) == 0) {
            return true;
        }
    }
    const char * layered[] = {
        "kq_soft_max_ext",
        "attn_norm",
        "attn_sub_norm",
        "attn_o_out",
        "Qcur",
        "Kcur",
        "Vcur",
        "ffn_inp",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_act",
        "ffn_down",
        "ffn_sub_norm",
        "ffn_out",
        "l_out",
    };
    for (const char * base : layered) {
        if (matches_layered_tensor(name, base)) {
            return true;
        }
    }
    return false;
}

void print_tensor_stats(const char * name, const ggml_tensor * t) {
    if (t == nullptr || name == nullptr) {
        return;
    }
    if (t->type != GGML_TYPE_F32) {
        return;
    }
    const int64_t n = ggml_nelements(t);
    if (n <= 0) {
        return;
    }
    const float * data = ggml_get_data_f32(t);
    if (data == nullptr) {
        return;
    }
    float min_v = data[0];
    float max_v = data[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double mean = sum / static_cast<double>(n);
    const double rms = std::sqrt(sum_sq / static_cast<double>(n));
    std::printf("DEBUG name=%s n=%lld min=%.9g max=%.9g mean=%.9g rms=%.9g\n",
        name, static_cast<long long>(n), min_v, max_v, mean, rms);
    if (g_print_values && name_values_matches(name)) {
        int count = g_values_n;
        if (count > n) {
            count = static_cast<int>(n);
        }
        std::printf("DEBUG_VALUES name=%s values=", name);
        for (int i = 0; i < count; ++i) {
            if (i > 0) std::printf(",");
            std::printf("%.9g", data[i]);
        }
        std::printf("\n");
    }
}

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = reinterpret_cast<DebugState *>(user_data);
    if (state == nullptr || !state->enabled) {
        return false;
    }
    const char * name = ggml_get_name(t);
    if (ask) {
        if (!name_matches(name)) {
            return false;
        }
        if (state->target_pos >= 0 && state->current_pos != state->target_pos) {
            return false;
        }
        return true;
    }
    if (!name_matches(name)) {
        return true;
    }
    if (state->target_pos >= 0 && state->current_pos != state->target_pos) {
        return true;
    }
    if (!state->seen.insert(name).second) {
        return true;
    }
    print_tensor_stats(name, t);
    if (state->i2s_dot && !state->i2s_dot_done && std::strcmp(name, "ffn_norm-0") == 0) {
        if (t->type == GGML_TYPE_F32 && t->ne[0] > 0 && state->model != nullptr) {
            ggml_tensor * w = llama_get_model_tensor(state->model, state->i2s_dot_tensor.c_str());
            if (w == nullptr) {
                std::printf("I2S_DOT error=missing_tensor name=%s\n", state->i2s_dot_tensor.c_str());
            } else if (w->type != GGML_TYPE_I2_S) {
                std::printf("I2S_DOT error=wrong_type name=%s type=%d\n", state->i2s_dot_tensor.c_str(), (int)w->type);
            } else if (state->i2s_dot_row < 0 || state->i2s_dot_row >= w->ne[1]) {
                std::printf("I2S_DOT error=row_oob name=%s row=%d rows=%lld\n", state->i2s_dot_tensor.c_str(), state->i2s_dot_row, (long long)w->ne[1]);
            } else {
                const int n = (int)w->ne[0];
                const float * x = ggml_get_data_f32(t);
                if (x != nullptr && n > 0) {
                    std::vector<int8_t> q((size_t)n);
                    float act_scale = 0.0f;
                    int32_t act_sum = 0;
                    quantize_row_i8_s(x, q.data(), n, &act_scale, &act_sum);

                    const size_t row_stride = (size_t)w->nb[1] / 4;
                    const uint8_t * wdata = (const uint8_t *)w->data;
                    const uint8_t * row_ptr = wdata + (size_t)state->i2s_dot_row * row_stride;

                    float dot = 0.0f;
                    ggml_vec_dot_i2_i8_s(n, &dot, 0, row_ptr, 0, q.data(), 0, 1);

                    const size_t scale_off = (size_t)w->ne[0] * (size_t)w->ne[1] / 4;
                    const float * wscale = (const float *)(wdata + scale_off);
                    const float weight_scale = wscale[0];

                    const float out = (dot - (float)act_sum) / act_scale * weight_scale;
                    std::printf("I2S_DOT name=%s row=%d dot=%.9g act_scale=%.9g act_sum=%d weight_scale=%.9g out=%.9g\n",
                        state->i2s_dot_tensor.c_str(), state->i2s_dot_row, dot, act_scale, act_sum, weight_scale, out);
                }
            }
        }
        state->i2s_dot_done = true;
    }
    return true;
}

} // namespace

int main() {
    const std::string model_path = env_or("BITNET_REF_MODEL", "");
    const std::string prompt = env_or("BITNET_REF_PROMPT", "");
    const int max_tokens = env_or_int("BITNET_REF_MAX_TOKENS", 32);
    const int topk = env_or_int("BITNET_REF_TOPK", 5);
    const int n_threads = env_or_int("BITNET_REF_THREADS", 0);
    const int n_ctx_override = env_or_int("BITNET_REF_N_CTX", 0);
    const bool token_by_token = env_or_bool("BITNET_REF_TOKEN_BY_TOKEN", false);
    g_print_values = env_or_bool("BITNET_REF_DEBUG_VALUES", false);
    g_values_n = env_or_int("BITNET_REF_DEBUG_VALUES_N", 8);
    g_values_name = env_or("BITNET_REF_DEBUG_VALUES_NAME", "");

    if (model_path.empty()) {
        std::fprintf(stderr, "BITNET_REF_MODEL is required\n");
        return 1;
    }
    if (max_tokens < 0) {
        std::fprintf(stderr, "BITNET_REF_MAX_TOKENS must be >= 0\n");
        return 1;
    }

    llama_log_set(silent_log_callback, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    DebugState debug_state;
    debug_state.enabled = env_or_bool("BITNET_REF_DEBUG", false);
    debug_state.target_pos = env_or_int("BITNET_REF_DEBUG_POS", -1);
    debug_state.i2s_dot = env_or_bool("BITNET_REF_I2S_DOT", false);
    debug_state.i2s_dot_tensor = env_or("BITNET_REF_I2S_DOT_TENSOR", "blk.0.ffn_gate.weight");
    debug_state.i2s_dot_row = env_or_int("BITNET_REF_I2S_DOT_ROW", 0);
    if (debug_state.enabled) {
        cparams.cb_eval = eval_callback;
        cparams.cb_eval_user_data = &debug_state;
    }
    if (n_ctx_override > 0) {
        cparams.n_ctx = static_cast<uint32_t>(n_ctx_override);
    } else {
        cparams.n_ctx = 0;
    }
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;

    llama_model * model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    debug_state.model = model;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create llama context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    if (n_threads > 0) {
        llama_set_n_threads(ctx, n_threads, n_threads);
    }

    const int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_ctx));
    int n_prompt = llama_tokenize(
        model,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        true,
        false);

    if (n_prompt < 0) {
        n_prompt = -n_prompt;
        prompt_tokens.resize(static_cast<size_t>(n_prompt));
        n_prompt = llama_tokenize(
            model,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            false);
    }
    if (n_prompt <= 0) {
        std::fprintf(stderr, "tokenization failed\n");
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    prompt_tokens.resize(static_cast<size_t>(n_prompt));
    for (int i = 0; i < n_prompt; ++i) {
        std::printf("PROMPT_TOKEN idx=%d id=%d\n", i, static_cast<int>(prompt_tokens[static_cast<size_t>(i)]));
    }

    if (token_by_token) {
        for (int i = 0; i < n_prompt; ++i) {
            llama_batch prompt_batch = llama_batch_init(1, 0, 1);
            prompt_batch.n_tokens = 1;
            prompt_batch.token[0] = prompt_tokens[static_cast<size_t>(i)];
            prompt_batch.pos[0] = i;
            prompt_batch.n_seq_id[0] = 1;
            prompt_batch.seq_id[0][0] = 0;
            prompt_batch.logits[0] = (i == n_prompt - 1) ? 1 : 0;
            debug_state.current_pos = i;
            debug_state.seen.clear();
            if (llama_decode(ctx, prompt_batch) != 0) {
                std::fprintf(stderr, "prompt decode failed\n");
                llama_batch_free(prompt_batch);
                llama_free(ctx);
                llama_free_model(model);
                llama_backend_free();
                return 1;
            }
            llama_batch_free(prompt_batch);
        }
    } else {
        llama_batch prompt_batch = llama_batch_init(n_prompt, 0, 1);
        prompt_batch.n_tokens = n_prompt;
        for (int i = 0; i < n_prompt; ++i) {
            prompt_batch.token[i] = prompt_tokens[static_cast<size_t>(i)];
            prompt_batch.pos[i] = i;
            prompt_batch.n_seq_id[i] = 1;
            prompt_batch.seq_id[i][0] = 0;
            prompt_batch.logits[i] = (i == n_prompt - 1) ? 1 : 0;
        }
        debug_state.current_pos = n_prompt - 1;
        debug_state.seen.clear();
        if (llama_decode(ctx, prompt_batch) != 0) {
            std::fprintf(stderr, "prompt decode failed\n");
            llama_batch_free(prompt_batch);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        llama_batch_free(prompt_batch);
    }

    const int n_vocab = llama_n_vocab(model);
    int pos = n_prompt;

    for (int step = 0; step < max_tokens; ++step) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "missing logits at step %d\n", step);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }

        std::vector<TopKEntry> entries = topk_from_logits(logits, n_vocab, topk);
        if (entries.empty()) {
            std::fprintf(stderr, "empty top-k at step %d\n", step);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }

        const llama_token token = entries[0].id;
        std::printf("TOKEN step=%d id=%d\n", step, token);
        print_topk_line(step, entries);

        double step_ms = 0.0;
        if (step + 1 < max_tokens) {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.n_tokens = 1;
            b.token[0] = token;
            b.pos[0] = pos++;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;

            debug_state.current_pos = b.pos[0];
            debug_state.seen.clear();
            const auto t0 = std::chrono::high_resolution_clock::now();
            const int rc = llama_decode(ctx, b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            llama_batch_free(b);
            if (rc != 0) {
                std::fprintf(stderr, "decode failed at step %d\n", step);
                llama_free(ctx);
                llama_free_model(model);
                llama_backend_free();
                return 1;
            }

            step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::printf("TIME step=%d ms=%.6f\n", step, step_ms);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
