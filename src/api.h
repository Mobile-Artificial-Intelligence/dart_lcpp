#include "llama.h"

struct p_sampling_params {
    float p;
    size_t min_keep;
};

struct temperature_sampling_params {
    float temperature;
    float delta;
    float exponent;
};

struct xtc_sampling_params {
    float probability;
    float threshold;
    size_t min_keep;
    uint32_t seed;
};

struct mirostat_sampling_params {
    int32_t n_vocab;
    uint32_t seed;
    float tau;
    float eta;
    int32_t m;
};

struct mirostat_v2_sampling_params {
    uint32_t seed;
    float tau;
    float eta;
};

struct grammar_sampling_params {
    const char * str;
    const char * root;
};

struct grammar_lazy_sampling_params {
    const char * str;
    const char * root;
    const char ** trigger_words;
    size_t num_trigger_words;
    const llama_token * trigger_tokens;
    size_t num_trigger_tokens;
};

struct penalties_sampling_params {
    int32_t last_n;
    float repeat;
    float freq;
    float present;
};

struct dry_sampling_params {
    int32_t n_ctx_train;
    float multiplier;
    float base;
    int32_t allowed_length;
    int32_t penalty_last_n;
    const char ** breakers;
    size_t num_breakers;
};

struct logit_bias_sampling_params {
    int32_t n_vocab;
    int32_t n_logit_bias;
    llama_logit_bias * logit_bias;
};

struct api_params {
    char * model_path;

    // llama_model_params
    bool vocab_only;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;

    // llama_context_params
    int n_ctx;
    int n_batch;
    int n_ubatch;
    int n_seq_max;
    int n_threads;
    int n_threads_batch;

    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_attention_type attention_type;

    double rope_freq_base;
    double rope_freq_scale;

    double yarn_ext_factor;
    double yarn_attn_factor;
    double yarn_beta_fast;
    double yarn_beta_slow;
    int yarn_orig_ctx;

    double defrag_thold;

    enum ggml_type type_k;
    enum ggml_type type_v;

    bool logits_all;
    bool embeddings;
    bool offload_kqv;
    bool flash_attn;
    bool no_perf;

    // llama_sampling
    bool greedy;
    bool infill;
    int seed;
    int top_k;
    struct p_sampling_params * top_p;
    struct p_sampling_params * min_p;
    struct p_sampling_params * typical_p;
    struct temperature_sampling_params * temperature;
    struct xtc_sampling_params * xtc;
    struct mirostat_sampling_params * mirostat;
    struct mirostat_v2_sampling_params * mirostat_v2;
    struct grammar_sampling_params * grammar;
    struct grammar_lazy_sampling_params * grammar_lazy;
    struct penalties_sampling_params * penalties;
    struct dry_sampling_params * dry;
    struct logit_bias_sampling_params * logit_bias;
};

LLAMA_API struct api_params api_default_params(void);

LLAMA_API int api_init(struct api_params params);

LLAMA_API int api_prompt(struct llama_chat_message * msg, size_t n_msg);

LLAMA_API void api_stop(void);

LLAMA_API void api_free(void);