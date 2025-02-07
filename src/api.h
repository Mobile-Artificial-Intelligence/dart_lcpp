#include "llama.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct p_sampling_params {
    float p;
    int min_keep;
};

struct temperature_sampling_params {
    float temperature;
    float delta;
    float exponent;
};

struct xtc_sampling_params {
    float probability;
    float threshold;
    int min_keep;
    unsigned int seed;
};

struct mirostat_sampling_params {
    int n_vocab;
    unsigned int seed;
    float tau;
    float eta;
    int m;
};

struct mirostat_v2_sampling_params {
    unsigned int seed;
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
    int num_trigger_words;
    const llama_token * trigger_tokens;
    int num_trigger_tokens;
};

struct penalties_sampling_params {
    int last_n;
    float repeat;
    float freq;
    float present;
};

struct dry_sampling_params {
    int n_ctx_train;
    float multiplier;
    float base;
    int allowed_length;
    int penalty_last_n;
    const char ** breakers;
    int num_breakers;
};

struct api_params {
    char * model_path;

    // llama_model_params
    bool vocab_only;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;

    // llama_context_params
    unsigned int n_ctx;
    unsigned int n_batch;
    unsigned int n_ubatch;
    unsigned int n_seq_max;
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
    unsigned int yarn_orig_ctx;

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
    unsigned int seed;
    int top_k;
    struct p_sampling_params top_p;
    struct p_sampling_params min_p;
    struct p_sampling_params typical_p;
    struct temperature_sampling_params temperature;
    struct xtc_sampling_params xtc;
    struct mirostat_sampling_params mirostat;
    struct mirostat_v2_sampling_params mirostat_v2;
    struct grammar_sampling_params grammar;
    struct grammar_lazy_sampling_params grammar_lazy;
    struct penalties_sampling_params penalties;
    struct dry_sampling_params dry;
};

LLAMA_API struct api_params api_default_params(void);

LLAMA_API int api_init(struct api_params params);

LLAMA_API int api_prompt(llama_chat_message * msg, size_t n_msg);

LLAMA_API void api_stop(void);

LLAMA_API void api_free(void);

#ifdef __cplusplus
}
#endif