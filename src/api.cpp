#include "api.h"

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
        /*.seed                     =*/ LLAMA_DEFAULT_SEED,
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