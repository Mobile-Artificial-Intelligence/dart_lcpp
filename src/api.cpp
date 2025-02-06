#include "api.h"

static llama_model * model;
static llama_context * ctx;
static llama_sampler * smpl;

struct api_params api_default_params() {
    auto default_model_params = llama_model_default_params();
    auto default_context_params = llama_context_default_params();

    struct api_params result = {
        /*.model_path               =*/ nullptr,
        /*.vocab_only               =*/ default_model_params.vocab_only,
        /*.use_mmap                 =*/ default_model_params.use_mmap,
        /*.use_mlock                =*/ default_model_params.use_mlock,
        /*.check_tensors            =*/ default_model_params.check_tensors,
        
        /*.n_ctx                    =*/ default_context_params.n_ctx,
        /*.n_batch                  =*/ default_context_params.n_batch,
        /*.n_ubatch                 =*/ default_context_params.n_ubatch,
        /*.n_seq_max                =*/ default_context_params.n_seq_max,
        /*.n_threads                =*/ default_context_params.n_threads,
        /*.n_threads_batch          =*/ default_context_params.n_threads_batch,
        /*.rope_scaling_type        =*/ default_context_params.rope_scaling_type,
        /*.pooling_type             =*/ default_context_params.pooling_type,
        /*.attention_type           =*/ default_context_params.attention_type,
        /*.rope_freq_base           =*/ default_context_params.rope_freq_base,
        /*.rope_freq_scale          =*/ default_context_params.rope_freq_scale,
        /*.yarn_ext_factor          =*/ default_context_params.yarn_ext_factor,
        /*.yarn_attn_factor         =*/ default_context_params.yarn_attn_factor,
        /*.yarn_beta_fast           =*/ default_context_params.yarn_beta_fast,
        /*.yarn_beta_slow           =*/ default_context_params.yarn_beta_slow,
        /*.yarn_orig_ctx            =*/ default_context_params.yarn_orig_ctx,
        /*.defrag_thold             =*/ default_context_params.defrag_thold,
        /*.type_k                   =*/ default_context_params.type_k,
        /*.type_v                   =*/ default_context_params.type_v,
        /*.logits_all               =*/ default_context_params.logits_all,
        /*.embeddings               =*/ default_context_params.embeddings,
        /*.offload_kqv              =*/ default_context_params.offload_kqv,
        /*.flash_attn               =*/ default_context_params.flash_attn,
        /*.no_perf                  =*/ default_context_params.no_perf,

        /*.greedy                   =*/ false,
        /*.infill                   =*/ false,
        /*.seed                     =*/ NULL,
        /*.top_k                    =*/ NULL,
        /*.top_p                    =*/ NULL,
        /*.min_p                    =*/ NULL,
        /*.typical_p                =*/ NULL,
        /*.temperature              =*/ NULL,
        /*.xtc                      =*/ NULL,
        /*.mirostat                 =*/ NULL,
        /*.mirostat_v2              =*/ NULL,
        /*.grammar                  =*/ NULL,
        /*.grammar_lazy             =*/ NULL,
        /*.penalties                =*/ NULL,
        /*.dry                      =*/ NULL,
        /*.logit_bias               =*/ NULL
    };

    return result;
}

int api_init(struct api_params params) {
    ggml_backend_load_all();

    auto model_params = llama_model_default_params();
    model_params.vocab_only = params.vocab_only;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;
    model_params.check_tensors = params.check_tensors;

    model = llama_model_load_from_file(params.model_path, model_params);

    auto context_params = llama_context_default_params();
    context_params.n_ctx = params.n_ctx;
    context_params.n_batch = params.n_batch;
    context_params.n_ubatch = params.n_ubatch;
    context_params.n_seq_max = params.n_seq_max;
    context_params.n_threads = params.n_threads;
    context_params.n_threads_batch = params.n_threads_batch;
    context_params.rope_scaling_type = params.rope_scaling_type;
    context_params.pooling_type = params.pooling_type;
    context_params.attention_type = params.attention_type;
    context_params.rope_freq_base = params.rope_freq_base;
    context_params.rope_freq_scale = params.rope_freq_scale;
    context_params.yarn_ext_factor = params.yarn_ext_factor;
    context_params.yarn_attn_factor = params.yarn_attn_factor;
    context_params.yarn_beta_fast = params.yarn_beta_fast;
    context_params.yarn_beta_slow = params.yarn_beta_slow;
    context_params.yarn_orig_ctx = params.yarn_orig_ctx;
    context_params.defrag_thold = params.defrag_thold;
    context_params.type_k = params.type_k;
    context_params.type_v = params.type_v;
    context_params.logits_all = params.logits_all;
    context_params.embeddings = params.embeddings;
    context_params.offload_kqv = params.offload_kqv;
    context_params.flash_attn = params.flash_attn;
    context_params.no_perf = params.no_perf;

    ctx = llama_init_from_model(model, context_params);

    auto vocab = llama_model_get_vocab(model);
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    if (params.greedy) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    if (params.infill) {
        llama_sampler_chain_add(smpl, llama_sampler_init_infill(vocab));
    }

    if (params.seed != NULL) {
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));
    }

    if (params.top_k != NULL) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    }

    if (params.top_p.p != NULL && params.top_p.min_keep != NULL) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p.p, params.top_p.min_keep));
    }

    if (params.min_p.p != NULL && params.min_p.min_keep != NULL) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params.min_p.p, params.min_p.min_keep));
    }

    if (params.typical_p.p != NULL && params.typical_p.min_keep != NULL) {
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(params.typical_p.p, params.typical_p.min_keep));
    }

    if (params.temperature.temperature != NULL) {
        if (params.temperature.delta != NULL && params.temperature.exponent != NULL) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(params.temperature.temperature, params.temperature.delta, params.temperature.exponent));
        } 
        else {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature.temperature));
        }
    }
}