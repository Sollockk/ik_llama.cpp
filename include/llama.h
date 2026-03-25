//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 9

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 3

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_model;
    struct llama_context;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    // pre-tokenization types
    //enum llama_vocab_pre_type {
    //    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
    //    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
    //    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
    //    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
    //    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
    //    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
    //    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
    //    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
    //    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
    //    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
    //    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
    //    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
    //    LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
    //    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
    //    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
    //    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
    //    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
    //    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
    //    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
    //    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28, //llama.cpp lists this as 28
    //    LLAMA_VOCAB_PRE_TYPE_GPT4O          = 29,
    //    LLAMA_VOCAB_PRE_TYPE_SUPERBPE       = 30,
    //    LLAMA_VOCAB_PRE_TYPE_TRILLION       = 31,
    //    LLAMA_VOCAB_PRE_TYPE_BAILINGMOE     = 32,
    //    LLAMA_VOCAB_PRE_TYPE_LLAMA4         = 33,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON_3       = 34,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON_E       = 35,
    //    LLAMA_VOCAB_PRE_TYPE_SEED_CODER     = 36, //llama.cpp lists this as 35
    //    LLAMA_VOCAB_PRE_TYPE_HUNYUAN        = 37, //llama.cpp lists this as 36
    //    LLAMA_VOCAB_PRE_TYPE_KIMI_K2        = 38, //llama.cpp lists this as 37
    //};

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4         = 38, // except 1d tensors, 38 to be compatible with mainline
        //
        LLAMA_FTYPE_MOSTLY_Q6_0          = 135, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_BN        = 136, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_BN        = 137, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_K         = 138, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_K         = 139, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_K         = 140, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_K         = 141, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ6_K         = 142, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KS        = 145, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KL        = 146, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KS        = 147, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KSS       = 148, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_KV         = 149, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_KS        = 150, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KT        = 151, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KT        = 152, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KT        = 153, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KS        = 154, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KL        = 155, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_KT        = 156, // except 1d tensors
                                                //
        LLAMA_FTYPE_MOSTLY_Q4_0_R8       = 202, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_0_R8       = 207, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0_R4       = 208, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_R4       = 210, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_R4       = 211, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_R4       = 214, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_R4       = 216, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K_R4       = 218, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4    = 219, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS_R4     = 220, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4    = 223, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S_R4      = 224, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL_R4     = 225, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S_R4      = 226, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M_R4      = 229, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS_R8     = 230, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M_R4      = 231, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_0_R4       = 335, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16_R16      = 232, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_BN_R4     = 337, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_K_R4      = 338, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_K_R4      = 339, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_K_R4      = 340, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_K_R4      = 341, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KS_R4     = 345, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_KS_R4     = 350, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_KV_R8      = 398, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_K_R8       = 399, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED = 1,
    };

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE    = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER   = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ATTN    = 2, // splits self-attention computations across GPUs
        LLAMA_SPLIT_MODE_GRAPH   = 3, // splits computations across GPUs
    };

    enum llama_mtp_op_type {
        MTP_OP_NONE             = 0,
        MTP_OP_WARMUP           = 1,
        MTP_OP_UPDATE_ACCEPTED  = 2,
        MTP_OP_DRAFT_GEN        = 3,
    };

    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    // - seq_id : the sequence to which the respective token belongs
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits; // TODO: rename this to "output"

        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        llama_pos    all_pos_0;  // used if pos == NULL
        llama_pos    all_pos_1;  // used if pos == NULL
        llama_seq_id all_seq_id; // used if seq_id == NULL
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_params {
        // comma separated list of devices to use for offloading
        const char* devices;

        int32_t n_gpu_layers; // number of layers to store in VRAM
        int32_t mla;          // MLA implementation to use (only applicable to DeepSeek models at this point)
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // main_gpu interpretation depends on split_mode:
        // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
        // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
        // LLAMA_SPLIT_LAYER: ignored
        int32_t main_gpu;
        int32_t max_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // comma separated list of RPC servers to use for offloading
        const char * rpc_servers;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;    // only load the vocabulary, no weights
        bool use_mmap;      // use mmap if possible
        bool use_mlock;     // force system to keep model in RAM
        bool check_tensors; // validate model tensor data
        bool repack_tensors;// repack if available
        bool use_thp;       // use transparent huge pages (linux only)
        bool validate_quants; // if true, check for NaNs while loading the model
        bool merge_qkv;     // if true, merge separate Q, K, V tensors into a single, contiguous tensor
        bool merge_up_gate_exps;  // if true, merge ffn_up_exps and ffn_gate_exps tensors into a single, contiguous tensor
        bool mtp;           // if true, load MTP layers if present
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggerganov/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t seed;              // RNG seed, -1 for random
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_threads;         // number of threads to use for generation
        uint32_t n_threads_batch;   // number of threads to use for batch processing
        int32_t  max_extra_alloc;   // Max. additional VRAM the scheduler is allowed to allocate

        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]
        enum ggml_type type_reduce; // data type for reduce operations

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
        bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
        int  mla_attn;    // whether to use MLA attention [EXPERIMENTAL]
        int  attn_max_batch;    // maximum batch size for attention computations [EXPERIMENTAL]
        bool fused_moe_up_gate; // whether to use fused MoE up/gate op
        bool grouped_expert_routing; // whether to use grouped expert routing (BailingMoeV2 arch)
        bool fused_up_gate;     // whether to use fused up/gate op [EXPERIMENTAL]
        bool fused_mmad;        // whether to use fused mul+multi_add op [EXPERIMENTAL]
        bool rope_cache;        // whether to use RoPE cache [EXPERIMENTAL]
        bool graph_reuse;       // whether to reuse graphs when possible [EXPERIMENTAL]
        int  min_experts;
        float thresh_experts;
        bool only_active_experts;
        bool k_cache_hadamard;  // if true, apply Hadamard transfrom to K-cache
        bool split_mode_graph_scheduling; // if true, force split mode graph scheduling
        //bool split_mode_f16;    // if true, cast intermediate results to f16 before copying to other GPUs
        bool scheduler_async;   // if true, with split mode "graph" graph evaluation will be done using multiple threads
        bool mtp;   // Activate MTP if supported
        enum llama_mtp_op_type mtp_op_type;

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
        void *              offload_policy;
        void *              cuda_params;
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                     // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;              // quantize to this llama_ftype
        enum ggml_type output_tensor_type;   // output tensor type
        enum ggml_type token_embedding_type; // token embeddings tensor type
        enum ggml_type attn_q_type;          // attention query tensor type
        enum ggml_type attn_k_type;          // attention key tensor type
        enum ggml_type attn_v_type;          // attention value tensor type
        enum ggml_type attn_qkv_type;        // attention query-key-value tensor type
        enum ggml_type attn_output_type;     // attention output tensor type
        enum ggml_type ffn_gate_type;        // feedforward network gate type
        enum ggml_type ffn_down_type;        // feedforward network down type
        enum ggml_type ffn_up_type;          // feedforward network up type
        enum ggml_type ffn_gate_inp_type;    // routed experts probabilities typy (relevant for MoE models only)
        bool allow_requantize;               // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;         // quantize output.weight
        bool only_copy;                      // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                           // quantize all tensors to the default type
        bool keep_split;                     // quantize to the same number of shards
        bool ignore_imatrix_rules;           // If set to true, the built-in rules for refusing to quantize into certain quants without imatrix are ignored
        bool only_repack;                    // Only repack tensors
        bool dry_run;                        //
        bool partial_requant;                // quantize only missing split files in the split quantized .gguf destination directory
        void * imatrix;                      // pointer to importance matrix data
        void * kv_overrides;                 // pointer to vector containing overrides
        void * custom_quants;                // pointer to vector containing custom quantization rules
        void * repack_pattern;               // pointer to a vector containing regexes to be used for matching tensor names. Can be null
    } llama_model_quantize_params;

    // grammar types
    struct llama_grammar;



    // performance timing information
    struct llama_timings {
        double t_start_ms;
        double t_end_ms;
        double t_load_ms;
        double t_sample_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_sample;
        int32_t n_p_eval;
        int32_t n_eval;
    };

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_lora_adapter;

    // Helpers for getting default parameters
    LLAMA_API struct llama_model_params llama_model_default_params(void);
    LLAMA_API struct llama_context_params llama_context_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
            struct llama_model_params     params);

    LLAMA_API void llama_free_model(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    LLAMA_API void llama_set_offload_policy(struct llama_context * lctx, int op, bool on_or_off);

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);



    LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    LLAMA_API enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);

    struct llama_vocab;
    LLAMA_API enum llama_vocab_type   llama_vocab_type(const struct llama_vocab * vocab);
    LLAMA_API enum llama_rope_type    llama_rope_type   (const struct llama_model * model);

    LLAMA_API const struct llama_vocab* llama_model_get_vocab(const struct llama_model* model);
    LLAMA_API const char* llama_model_chat_template(const struct llama_model* model, const char* name);
    LLAMA_API int32_t llama_n_vocab    (const struct llama_model * model);
    LLAMA_API const struct llama_vocab* llama_get_model_vocab(const struct llama_model* model);
    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
    LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp(const struct llama_model* model);

    LLAMA_API int32_t llama_n_layer    (const struct llama_model * model);

    // Override the number of experts used per token for MoE models.
    // 0 = use model default.  Reduces compute by n_expert_used_old/n_expert_used_new.
    LLAMA_API void llama_model_set_n_expert_used(struct llama_model * model, int32_t n_expert_used);

    // Compat
    LLAMA_API bool        llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool        llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API int32_t     llama_vocab_n_tokens(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Get a llama model tensor
    LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);

    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    LLAMA_API bool llama_model_has_recurrent(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    // Load a LoRA adapter from file
    // The loaded adapter will be associated to the given model, and will be free when the model is deleted
    LLAMA_API struct llama_lora_adapter * llama_lora_adapter_init(
            struct llama_model * model,
            const char * path_lora);

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    LLAMA_API int32_t llama_lora_adapter_set(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    LLAMA_API int32_t llama_lora_adapter_remove(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter);

    // Remove all LoRA adapters from given context
    LLAMA_API void llama_lora_adapter_clear(
            struct llama_context * ctx);

    // Manually free a LoRA adapter
    // Note: loaded adapters will be free when the associated model is deleted
    LLAMA_API void llama_lora_adapter_free(struct llama_lora_adapter * adapter);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_control_vector_apply(
            struct llama_context * lctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // KV cache
    //

    // Information associated with an individual cell in the KV cache view.
    struct llama_kv_cache_view_cell {
        // The position for this cell. Takes KV cache shifts into account.
        // May be negative if the cell is not populated.
        llama_pos pos;
    };

    // An updateable view of the KV cache.
    struct llama_kv_cache_view {
        // Number of KV cache cells. This will be the same as the context size.
        int32_t n_cells;

        // Maximum number of sequences that can exist in a cell. It's not an error
        // if there are more sequences in a cell than this value, however they will
        // not be visible in the view cells_sequences.
        int32_t n_seq_max;

        // Number of tokens in the cache. For example, if there are two populated
        // cells, the first with 1 sequence id in it and the second with 2 sequence
        // ids then you'll have 3 tokens.
        int32_t token_count;

        // Number of populated cache cells.
        int32_t used_cells;

        // Maximum contiguous empty slots in the cache.
        int32_t max_contiguous;

        // Index to the start of the max_contiguous slot range. Can be negative
        // when cache is full.
        int32_t max_contiguous_idx;

        // Information for an individual cell.
        struct llama_kv_cache_view_cell * cells;

        // The sequences for each cell. There will be n_seq_max items per cell.
        llama_seq_id * cells_sequences;
    };

    // work only with partial states, such as recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t llama_state_seq_flags;

    // Create an empty KV cache view. (use only for debugging purposes)
    LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);

    // Free a KV cache view. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_free(struct llama_kv_cache_view * view);

    // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);

    // Returns the number of tokens in the KV cache (slow, use only for debug)
    // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);

    // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);

    // Clear the KV cache - both cell info is erased and KV data is zeroed
    LLAMA_API void llama_kv_cache_clear(
            struct llama_context * ctx);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_kv_cache_seq_rm(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_cp(
            struct llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   p0,
                       llama_pos   p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_kv_cache_seq_keep(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_add(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta);

    // Integer division of the positions by factor of `d > 1`
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_div(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                             int   d);

    // Returns the largest position present in the KV cache for the specified sequence
    LLAMA_API llama_pos llama_kv_cache_seq_pos_max(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Returns the smallest position present in the KV cache for the specified sequence
    LLAMA_API llama_pos llama_kv_cache_seq_pos_min(
        struct llama_context * ctx,
        llama_seq_id   seq_id);

    // Defragment the KV cache
    // This will be applied:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    LLAMA_API void llama_kv_cache_defrag(struct llama_context * ctx);

    // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    // Positive return values does not mean a fatal error, but rather a warning.
    //    0 - success
    //    1 - Context overflow in a model where k-shift is not supported
    LLAMA_API int32_t llama_kv_cache_update(struct llama_context * ctx);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (rng, logits, embedding and kv_cache)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the KV cache of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    // Copy the KV cache of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens starting at pos_0
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens,
                    llama_pos   pos_0,
                 llama_seq_id   seq_id);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Processes a batch of tokens with the ecoder part of the encoder-decoder model.
    // Stores the encoder output internally for later use by the decoder cross-attention layers.
    //   0 - success
    // < 0 - error
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Positive return values does not mean a fatal error, but rather a warning.
    //   0 - success
    //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    // < 0 - error
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLAMA_API uint32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API uint32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the model is in embeddings mode or not
    // If true, embeddings will be returned but logits will not
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Layer skipping for "turbo" draft mode.
    // When set, the specified layers are completely skipped during graph
    // building (no attention, no FFN — the layer input passes straight
    // through).  This allows a fast "ultra-blurry" draft tier that runs
    // only a subset of layers:
    //   ultra-blurry (layer-skip) → blurry (all layers) → sharp (overlay)
    // Pass NULL/0 to clear (run all layers).
    LLAMA_API void llama_set_skip_layers(
            struct llama_context * ctx,
            const int32_t        * layer_indices,
            int32_t                n_layers);

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // Vocab
    //

    LLAMA_API const char * llama_token_get_text(const struct llama_model * model, llama_token token);

    LLAMA_API float llama_token_get_score(const struct llama_model * model, llama_token token);

    LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_model * model, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_token_is_eog(const struct llama_model * model, llama_token token);
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_token_is_control(const struct llama_model * model, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
    LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence
    LLAMA_API llama_token llama_token_cls(const struct llama_model * model); // classification
    LLAMA_API llama_token llama_token_sep(const struct llama_model * model); // sentence separator
    LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line
    LLAMA_API llama_token llama_token_pad(const struct llama_model * model); // padding

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t llama_add_bos_token(const struct llama_model * model);

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t llama_add_eos_token(const struct llama_model * model);

    // Codellama infill tokens
    LLAMA_API llama_token llama_token_prefix(const struct llama_model * model); // Beginning of infill prefix
    LLAMA_API llama_token llama_token_middle(const struct llama_model * model); // Beginning of infill middle
    LLAMA_API llama_token llama_token_suffix(const struct llama_model * model); // Beginning of infill suffix
    LLAMA_API llama_token llama_token_eot   (const struct llama_model * model); // End of infill middle

    //
    // Tokenization
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_model * model,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);
    LLAMA_API int32_t llama_vocab_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_model * model,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);
    LLAMA_API int32_t llama_token_to_piece_vocab(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of common_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    //LLAMA_API int32_t llama_detokenize(
    //    const struct llama_model * model,
    //           const llama_token * tokens,
    //                     int32_t   n_tokens,
    //                        char * text,
    //                     int32_t   text_len_max,
    //                        bool   remove_special,
    //                        bool   unparse_special);

    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
        const llama_token * tokens,
        int32_t   n_tokens,
        char * text,
        int32_t   text_len_max,
        bool   remove_special,
        bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
              const struct llama_model * model,
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);
    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    typedef void* llama_sampler_context_t;

    struct llama_sampler;

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char* (*name)  (const struct llama_sampler*);                               // can be NULL
        void                   (*accept)(struct llama_sampler*, llama_token);             // can be NULL
        void                   (*apply) (struct llama_sampler*, llama_token_data_array*); // required
        void                   (*reset) (struct llama_sampler*);                          // can be NULL
        struct llama_sampler* (*clone) (const struct llama_sampler*);                     // can be NULL if ctx is NULL
        void                   (*free)  (struct llama_sampler* smpl);                     // can be NULL if ctx is NULL
    };

    struct llama_sampler {
        struct llama_sampler_i* iface;
        llama_sampler_context_t   ctx;
    };

    //
    // Grammar
    //

    /// Initialize a llama_grammar.
    ///
    /// @param rules The rule elements of the grammar to initialize.
    /// @param n_rules The number of rules.
    /// @param start_rule_index The index of the root rule (the starting point of the grammar).
    /// @return The initialized llama_grammar or nullptr if initialization failed.
    //LLAMA_API struct llama_grammar * llama_grammar_init(
    //        const llama_grammar_element ** rules,
    //                             size_t    n_rules,
    //                             size_t    start_rule_index);

    struct llama_sampler_grammar;
    LLAMA_API void llama_grammar_init_lazy(struct llama_sampler_grammar * grammar);

    LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);

    LLAMA_API struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);

    /// @details Apply constraints from grammar
    LLAMA_API void llama_grammar_sample(
            const struct llama_grammar * grammar,
            const struct llama_context * ctx,
                llama_token_data_array * candidates);
    LLAMA_API DEPRECATED(void llama_sample_grammar(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
      const struct llama_grammar * grammar),
        "use llama_grammar_sample instead");

    /// @details Accepts the sampled token into the grammar
    LLAMA_API void llama_grammar_accept_token(
            struct llama_grammar * grammar,
            struct llama_context * ctx,
                     llama_token   token);

    //
    // Sampling functions
    //

    // Sets the current rng seed.
    LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    LLAMA_API void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present);

    /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    /// @param logits Logits extracted from the original generation context.
    /// @param logits_guidance Logits extracted from a separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    LLAMA_API void llama_sample_apply_guidance(
              struct llama_context * ctx,
                             float * logits,
                             float * logits_guidance,
                             float   scale);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_API void llama_sample_softmax(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    LLAMA_API void llama_sample_dist(
        struct llama_context * ctx,
        llama_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_k(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                         int32_t   k,
                          size_t   min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
    LLAMA_API void llama_sample_min_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    LLAMA_API void llama_sample_tail_free(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   z,
                          size_t   min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API void llama_sample_typical(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API void llama_sample_entropy(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   min_temp,
                           float   max_temp,
                           float   exponent_val);

    LLAMA_API void llama_sample_temp(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   temp);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API void llama_sample_xtc(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   probability,
                           float   threshold,
                           size_t  min_keep);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API void llama_sample_top_n_sigma(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   top_n_sigma);


LLAMA_API void                   llama_sampler_reset(struct llama_sampler* smpl);

/// @details Intializes a GBNF grammar, see grammars/README.md for details.
/// @param vocab The vocabulary that this grammar will be used with.
/// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
/// @param grammar_root The name of the start symbol for the grammar.
LLAMA_API struct llama_grammar* llama_sampler_init_grammar(
    const struct llama_vocab* vocab,
    const char* grammar_str,
        const char* grammar_root);

/// @details Lazy grammar sampler, introduced in https://github.com/ggerganov/llama.cpp/pull/9639
/// @param trigger_words A list of words that will trigger the grammar sampler. This may be updated to a loose regex syntax (w/ ^) in a near future.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler.
DEPRECATED(LLAMA_API struct llama_grammar* llama_sampler_init_grammar_lazy(
    const struct llama_vocab* vocab,
        const char* grammar_str,
        const char* grammar_root,
        const char** trigger_words,
        size_t num_trigger_words,
        const llama_token* trigger_tokens,
        size_t num_trigger_tokens),
    "use llama_sampler_init_grammar_lazy_patterns instead");


/// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
/// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
LLAMA_API struct llama_grammar* llama_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    const char** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens);

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler_dry * llama_sampler_init_dry(
        const struct llama_vocab* model,
        float    dry_multiplier,
        float    dry_base,
        int32_t    dry_allowed_length,
        int32_t    dry_penalty_last_n,
        const char** seq_breakers,
        size_t    num_breakers);

    //LLAMA_API void llama_sample_dry(struct llama_context* ctx, llama_token_data_array* candidates_p, int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers);

    void llama_sample_dry(struct llama_context* ctx, struct llama_sampler_dry* smpl, llama_token_data_array* candidates_p);

    void llama_sampler_dry_reset(struct llama_sampler_dry* smpl);

    void llama_sampler_dry_free(struct llama_sampler_dry* smpl);

    struct llama_sampler_dry* llama_sampler_dry_clone(struct llama_sampler_dry* smpl);

    void llama_sampler_dry_accept(struct llama_sampler_dry* smpl, llama_token token);

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982


    /// @details Adaptive p sampler initializer
    /// @param target Select tokens near this probability (valid range 0.0 to 1.0; <0 = disabled)
    /// @param decay Decay rate for target adaptation over time. lower values -> faster but less stable adaptation. (valid range 0.0 to 1.0; ≤0 = no adaptation)
    LLAMA_API struct llama_sampler_adaptive_p * llama_init_adaptive_p(int n_vocab,
           const float target,
           const float decay,
            const bool updt_w_cur,
        const uint32_t seed);

    void llama_prep_adaptive_p(struct llama_context * ctx,
                                  float * logits,
        struct llama_sampler_adaptive_p * adapt_p_ctx);

    /// @details Adaptive p sampler described in https://github.com/MrJackSpade/adaptive-p-docs/blob/main/README.md
    void llama_sample_adaptive_p(struct llama_context * ctx,
                               llama_token_data_array * candidates,
                      struct llama_sampler_adaptive_p * adapt_p_ctx);

    void llama_review_adaptive_p(struct llama_sampler_adaptive_p * adapt_p_ctx, const int32_t n_rewind);


    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                         int32_t   m,
                           float * mu);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat_v2(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                           float * mu);

    /// @details Selects the token with the highest probability.
    ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
    LLAMA_API llama_token llama_sample_token_greedy(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities using the RNG of ctx.
    LLAMA_API llama_token llama_sample_token(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Randonly selects a token from the candidates following adaptive p sampler.
    llama_token llama_sample_token_adaptive_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
 struct llama_sampler_adaptive_p * adapt_p_ctx);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Performance information
    LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);

    LLAMA_API void llama_print_timings(struct llama_context * ctx);
    LLAMA_API void llama_reset_timings(struct llama_context * ctx);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

    LLAMA_API void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);

    //
    // Blurry→Sharp overlay system
    //
    // Enables selective quality upgrading during inference: load a low-quality
    // (blurry) GGUF model for speed, then hotswap specific layer weights from
    // a high-quality (sharp) GGUF model on-the-fly based on routing decisions.
    //

    struct llama_blurry_sharp_context;

    enum llama_blurry_sharp_router_strategy {
        LLAMA_BS_ROUTER_ALWAYS = 0,   // always sharpen eligible layers
        LLAMA_BS_ROUTER_NEVER  = 1,   // never sharpen (useful for A/B testing)
        LLAMA_BS_ROUTER_NORM   = 2,   // activation L2-norm heuristic
    };

    enum llama_blurry_sharp_eviction_policy {
        LLAMA_BS_EVICT_LRU      = 0,
        LLAMA_BS_EVICT_FIFO     = 1,
        LLAMA_BS_EVICT_PRIORITY = 2,
    };

    struct llama_blurry_sharp_params {
        const char * sharp_model_path;          // path to the sharp (high-quality) GGUF file

        enum llama_blurry_sharp_router_strategy router_strategy; // default: ALWAYS
        float    router_min_confidence;         // threshold for NORM router [0,1], default 0.8
        int32_t  max_sharp_layers;              // 0 = unlimited
        int64_t  memory_budget_bytes;           // 0 = unlimited (total CPU+GPU backup budget)
        int64_t  gpu_budget_bytes;              // 0 = unlimited; max GPU memory the overlay may allocate
                                                // for sharp device buffers.  When exceeded, device tensors
                                                // are skipped (CPU tensors still use zero-copy mmap).
        bool     restore_after_forward;         // restore blurry weights after each layer
        enum llama_blurry_sharp_eviction_policy eviction_policy; // default: LRU

        const int32_t * layer_allowlist;        // NULL = all layers eligible
        int32_t         n_layer_allowlist;
        const int32_t * layer_denylist;         // NULL = no layers denied
        int32_t         n_layer_denylist;

        bool     use_mmap;                      // mmap the sharp file (recommended)
        bool     verbose;                       // detailed logging
        bool     retain_device_buffers;         // keep GPU buffers across restore cycles so
                                                // re-sharpening skips cudaMalloc + PCIe copy.
                                                // Uses more persistent VRAM but dramatically
                                                // speeds up repeated sharpen/restore cycles
                                                // (e.g. combined/speculative mode).

        bool     permanent;                     // one-way sharpen: overwrite blurry weights
                                                // in-place with no backup.  Cannot be restored.
                                                // For GPU tensors with same type/size: writes
                                                // sharp data directly into the existing device
                                                // buffer (zero extra VRAM).  For different
                                                // types: allocates a new buffer but does NOT
                                                // keep the blurry backup.
                                                // For CPU tensors: pointer-swaps to sharp mmap
                                                // and releases blurry pages immediately.
                                                // Use for static overlay (apply once at startup)
                                                // to avoid doubling memory usage.

        bool     lazy_swap;                     // lazy per-layer swap staging: instead of
                                                // pre-reading the entire sharp model into RAM
                                                // and staging to swap at startup (slow), lazily
                                                // populate the RAM cache per-layer on first
                                                // apply_layer() call, and MADV_PAGEOUT the
                                                // cached pages on restore.  Subsequent accesses
                                                // swap-in from SSD (fast) instead of re-reading
                                                // from the GGUF file (slow).
                                                // Eliminates the 30-minute startup cost of
                                                // --bs-stage-swap by spreading disk->RAM->swap
                                                // migration across the first inference pass.

        bool     retain_mmap_pages;             // keep mmap pages in page cache after reading
                                                // instead of calling MADV_DONTNEED.  When you
                                                // have plenty of free RAM this dramatically
                                                // speeds up repeated sharpen/restore cycles by
                                                // keeping sharp data in the OS page cache.
                                                // Without this, each apply reads from disk.
                                                // With this, only the first apply touches disk;
                                                // subsequent cycles hit the page cache (RAM).

        int32_t  n_sharp_experts_gpu;           // mixed-precision MoE (GPU/generation): only overlay
                                                // the top-N most-used experts per layer with sharp data.
                                                // -1 = overlay all selected experts (default).
                                                //  0 = no GPU expert uploads (CPU-only JIT overlay).
                                                // >0 = overlay only top-N most frequently selected.
                                                // E.g. with top-8 routing, n_sharp_experts_gpu=4
                                                // overlays the 4 most frequently selected experts
                                                // and leaves the rest at blurry quality.

        int32_t  n_sharp_experts_cpu;           // mixed-precision MoE (CPU/prompt): same as above
                                                // but used during prompt processing when MoE runs
                                                // on CPU.  CPU has more memory headroom so this can
                                                // be higher than n_sharp_experts_gpu.
                                                // -1 = overlay all selected experts (default).
                                                //  0 = skip JIT overlay entirely during prompt.
                                                // -2 = use same value as n_sharp_experts_gpu.

        bool     parallel_expert_io;            // use parallel pread threads for expert slice I/O
                                                // instead of sequential reads.  Improves throughput
                                                // on NVMe SSDs with warm page cache.

        int64_t  gpu_cache_bytes;               // GPU expert cache size in bytes (0 = disabled).
                                                // Pre-allocates a persistent GPU buffer to cache
                                                // sharp expert slices.  Cache hits use fast GPU→GPU
                                                // copy (~10μs) instead of SSD→host→GPU (~5ms).
                                                // Recommended: 4096-8192 MiB for 160-expert models.

        int64_t  ram_cache_bytes;               // RAM expert cache size in bytes (0 = auto 4 GiB).
                                                // Caches sharp expert slices in anonymous memory
                                                // (not file-backed mmap) so the kernel swaps them
                                                // to SSD swap instead of dropping them.  Swap-backed
                                                // pages are faster to recover than re-faulting from
                                                // a GGUF mmap.  Set -1 to disable.

        bool     flash_experts;                 // flash-moe style expert streaming: instead of loading
                                                // blurry expert weights into RAM and overlaying sharp
                                                // data on top, stream Q4_K_M expert slices directly
                                                // from SSD on demand.  Only the K active experts are
                                                // read per layer (via pread), and the OS page cache
                                                // naturally caches hot experts.  Benefits:
                                                //  - All expert compute at Q4_K_M quality (faster dequant)
                                                //  - No backup/restore overhead
                                                //  - Blurry expert data never loaded → ~14GB less RAM
                                                //  - Freed RAM becomes OS page cache for expert reads
    };

    struct llama_blurry_sharp_state {
        int32_t n_layers_total;
        int32_t n_layers_sharpened;
        int64_t total_backup_bytes;
        int64_t total_sharp_bytes_read;
        int64_t memory_budget_bytes;
        int64_t gpu_budget_bytes;               // configured GPU budget (0 = unlimited)
        int64_t gpu_device_bytes_used;          // current GPU memory allocated by overlay
        int32_t max_sharp_layers;
        int32_t n_device_tensors_skipped;       // tensors skipped due to GPU budget
    };

    // Get default blurry-sharp parameters
    LLAMA_API struct llama_blurry_sharp_params llama_blurry_sharp_default_params(void);

    // Initialize overlay system.  Opens the sharp GGUF, builds tensor index.
    // Returns NULL on failure (error is logged).
    LLAMA_API struct llama_blurry_sharp_context * llama_blurry_sharp_init(
            struct llama_model              * model,
            struct llama_blurry_sharp_params  params);

    // Free all resources owned by the blurry-sharp context.
    // Restores all layers to blurry weights before freeing.
    LLAMA_API void llama_blurry_sharp_free(
            struct llama_blurry_sharp_context * bsctx);

    // Apply sharp weights for a specific layer (backs up blurry data).
    // Returns 0 on success, negative on error.
    LLAMA_API int32_t llama_blurry_sharp_apply_layer(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx);

    // Restore a layer to its original blurry weights.
    // Returns 0 on success, -1 if layer was not sharpened.
    LLAMA_API int32_t llama_blurry_sharp_restore_layer(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx);

    // Restore all currently-sharpened layers.
    LLAMA_API void llama_blurry_sharp_restore_all(
            struct llama_blurry_sharp_context * bsctx);

    // Route + auto-sharpen in one call.  Returns number of layers sharpened.
    // activations: optional float[n_tokens * n_embd] for NORM router.
    LLAMA_API int32_t llama_blurry_sharp_auto_sharpen(
            struct llama_blurry_sharp_context * bsctx,
            const float                       * activations,
            int32_t                             n_tokens,
            int32_t                             n_embd);

    // Query whether a layer is currently sharpened.
    LLAMA_API bool llama_blurry_sharp_is_layer_sharp(
            const struct llama_blurry_sharp_context * bsctx,
            int32_t                                   layer_idx);

    // Get aggregate overlay state.
    LLAMA_API struct llama_blurry_sharp_state llama_blurry_sharp_get_state(
            const struct llama_blurry_sharp_context * bsctx);

    // Evict sharpened layers until backup bytes <= target_bytes.
    // Returns the number of layers evicted.
    LLAMA_API int32_t llama_blurry_sharp_evict_to_budget(
            struct llama_blurry_sharp_context * bsctx,
            int64_t                             target_bytes);

    // Apply all eligible layers in one call.
    // Prefetches mmap pages with madvise(WILLNEED) before starting,
    // reducing page-fault latency during the overlay loop.
    // Returns the number of layers successfully sharpened.
    LLAMA_API int32_t llama_blurry_sharp_apply_all(
            struct llama_blurry_sharp_context * bsctx);

    // Prefetch live zero-copy tensor pages into the page cache (RAM).
    // After sharpening, CPU zero-copy tensors point at mmap pages that may
    // not be resident in RAM yet.  This tells the kernel to pull them in
    // using available RAM, avoiding page-fault stalls during generation.
    // For GPU tensors this is a no-op (data is in VRAM).
    LLAMA_API void llama_blurry_sharp_warm_live_pages(
            struct llama_blurry_sharp_context * bsctx);

    // Prefetch a layer's sharp data from disk into the page cache (RAM).
    // Issues madvise(WILLNEED) on the layer's mmap regions, which tells the
    // kernel to begin asynchronous readahead.  This is non-blocking and
    // returns immediately.  Call this N layers ahead of apply_layer /
    // apply_experts to give the kernel time to bring pages into RAM before
    // they are needed.  Safe to call for any memory tier — if the layer
    // uses RAM cache instead of mmap, or if tensors are on GPU, this is
    // effectively a no-op for those tensors (the mmap prefetch still helps
    // for the source data that will be copied to staging/device buffers).
    LLAMA_API void llama_blurry_sharp_prefetch_layer(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx);

    // Prefetch specific expert slices for a layer into page cache.
    // Uses madvise(WILLNEED) on only the given expert IDs' data regions,
    // avoiding the cost of prefetching all 160 experts (~2GB) per layer.
    // Ideal for lookahead prefetch using the current layer's routing
    // as a prediction for upcoming layers.
    LLAMA_API void llama_blurry_sharp_prefetch_expert_slices(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx,
            const int32_t                     * expert_ids,
            int32_t                             n_experts);

    // Start async prefetch for the next layer's expert slices using parallel
    // pread in a background thread.  When apply_experts is called for the
    // prefetched layer, it consumes the pre-read data with zero I/O.
    LLAMA_API void llama_blurry_sharp_async_prefetch_start(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx,
            const int32_t                     * expert_ids,
            int32_t                             n_experts);

    // Prefetch multiple layers in parallel using n_threads worker threads.
    // Each worker reads its assigned layers' tensor data from mmap/file
    // into a staging cache concurrently.  The subsequent sequential apply
    // loop finds data already cached, avoiding I/O stalls.
    // n_threads <= 0 defaults to 4.
    LLAMA_API void llama_blurry_sharp_prefetch_layers_parallel(
            struct llama_blurry_sharp_context * bsctx,
            const int32_t                     * layer_indices,
            int32_t                             n_layers,
            int32_t                             n_threads);

    // Issue aggressive readahead for ALL sharp model data.
    // Tells the kernel to populate the page cache for the entire sharp
    // model.  Call at the START of a draft phase to give the kernel
    // maximum lead time before apply_experts needs the data.
    // Non-blocking — returns immediately, kernel reads asynchronously.
    LLAMA_API void llama_blurry_sharp_readahead_all(
            struct llama_blurry_sharp_context * bsctx);

    // Pre-allocate device (GPU) buffers for all eligible tensors.
    // Moves cudaMalloc out of the apply hot-path so that subsequent
    // apply_layer / apply_all calls only need the PCIe data copy.
    // Implicitly enables retain_device_buffers.
    // Returns the number of device buffers pre-allocated.
    LLAMA_API int32_t llama_blurry_sharp_preload_device_cache(
            struct llama_blurry_sharp_context * bsctx);

    // Temporarily inflate expert tensor types to the sharp type so that
    // the backend scheduler allocates device copies at the larger size.
    // Must be called AFTER graph build but BEFORE ggml_backend_sched_alloc_graph.
    // Call llama_blurry_sharp_deflate_expert_types() to restore after alloc.
    //
    // n_tokens: current batch size.  For small batches (generation), also
    //   shrinks ne[2] from n_expert to n_expert_used so device copies are
    //   tiny (~122 MiB vs ~2.5 GiB).  For large batches (prompt), only
    //   inflates the type (ne[2] stays at n_expert).
    LLAMA_API void llama_blurry_sharp_inflate_expert_types(
            struct llama_blurry_sharp_context * bsctx,
            int32_t n_tokens,
            const int32_t * priority_layers,
            int32_t         n_priority_layers);

    // Set a pre-allocated GPU buffer as the expert cache.
    // This should be called AFTER llama_blurry_sharp_init() but BEFORE inference.
    // The buffer must have been allocated on a GPU backend.
    // Takes ownership of the buffer (freed by llama_blurry_sharp_free).
    LLAMA_API void llama_blurry_sharp_set_gpu_cache_buffer(
            struct llama_blurry_sharp_context * bsctx,
            ggml_backend_buffer_t               buf,
            ggml_backend_t                      backend);

    // Restore expert tensor types to their original (blurry) types after
    // the scheduler has allocated device copies at the inflated size.
    LLAMA_API void llama_blurry_sharp_deflate_expert_types(
            struct llama_blurry_sharp_context * bsctx);

    // -----------------------------------------------------------------------
    // Memory-tier management: VRAM > RAM > Swap > Disk
    //
    // By default, CPU tensors use mmap zero-copy: their data pointers point
    // directly at file-backed mmap pages.  Under memory pressure, the kernel
    // DROPS these pages (they're clean/file-backed) and re-faults from disk.
    // This means the memory hierarchy is effectively: VRAM > Disk, skipping
    // both RAM and Swap tiers entirely.
    //
    // These functions populate a RAM cache of anonymous heap buffers.
    // Anonymous pages behave differently under pressure: instead of being
    // dropped, they are SWAPPED OUT to the swap partition.  Swap-in from
    // an SSD is typically 10-50x faster than random I/O from a GGUF file.
    //
    // Recommended startup sequence:
    //   1. llama_blurry_sharp_init()
    //   2. llama_blurry_sharp_precache_ram()    — fills RAM with sharp data
    //   3. llama_blurry_sharp_stage_to_swap()   — moves to swap, frees RAM
    //   4. llama_blurry_sharp_preload_device_cache()  — pre-alloc GPU buffers
    //   5. llama_blurry_sharp_apply_all() / apply_layer()
    //
    // After step 3, the sharp data lives in swap (fast SSD) and RAM is free
    // for the blurry model, KV cache, and other active data.  When apply
    // needs sharp data, it reads from the RAM cache (swap-backed) instead
    // of from the GGUF file on disk.
    // -----------------------------------------------------------------------

    // Pre-read all sharp tensor data into anonymous heap buffers (RAM cache).
    // This fills available RAM with sharp data.  Under memory pressure, these
    // anonymous pages are swapped out (not dropped like file-backed mmap pages).
    // Re-access comes from swap (fast SSD) instead of the GGUF file (slow).
    // Returns the number of tensors cached.
    LLAMA_API int32_t llama_blurry_sharp_precache_ram(
            struct llama_blurry_sharp_context * bsctx);

    // Proactively move RAM-cached pages to swap using MADV_PAGEOUT (Linux 5.4+).
    // This frees RAM for the blurry model / KV cache while keeping sharp data
    // quickly accessible via swap-in.  No-op on systems without MADV_PAGEOUT.
    // Returns the number of tensors whose pages were staged to swap.
    LLAMA_API int32_t llama_blurry_sharp_stage_to_swap(
            struct llama_blurry_sharp_context * bsctx);

    // -----------------------------------------------------------------------
    // MoE Combination Expert API
    //
    // For Mixture-of-Experts models, expert FFN tensors (ffn_gate_exps,
    // ffn_up_exps, ffn_down_exps) are 3D tensors where the outermost
    // dimension (ne[2]) is the expert index.  Instead of sharpening an
    // entire layer (which replaces ALL experts, including inactive ones),
    // these functions allow sharpening individual expert slices.
    //
    // This creates a "combination expert" tensor: mostly blurry (fast)
    // expert weights with selected expert slices replaced by sharp
    // (high-quality) data.  Only the active experts need sharpening,
    // reducing I/O by a factor of n_expert / n_expert_used.
    //
    // Non-expert tensors in the layer (attention, norms) are always
    // sharpened in full when apply_experts is called, since they are
    // shared across all experts and are comparatively small.
    // -----------------------------------------------------------------------

    // Apply sharp weights for specific expert(s) within a layer.
    // Creates a "combination tensor" per expert tensor: copies the full
    // blurry tensor, then overwrites only the requested expert slices
    // with sharp data.  Non-expert tensors (attn, norms) in the layer
    // are sharpened in full.
    //
    // expert_ids:   array of expert indices to sharpen
    // n_experts:    number of entries in expert_ids
    //
    // Returns 0 on success, negative on error.
    // If the layer has no expert tensors (dense model), falls back to
    // llama_blurry_sharp_apply_layer().
    LLAMA_API int32_t llama_blurry_sharp_apply_experts(
            struct llama_blurry_sharp_context * bsctx,
            int32_t                             layer_idx,
            const int32_t                     * expert_ids,
            int32_t                             n_experts);

    // Query the number of experts per MoE layer in the sharp model.
    // Returns 0 if the model is not MoE or if the context is invalid.
    // Inspects the ne[2] dimension of the first ffn_gate_exps tensor found.
    LLAMA_API int32_t llama_blurry_sharp_n_experts(
            const struct llama_blurry_sharp_context * bsctx);

    // Query the number of active (routed) experts per token.
    // Returns 0 if unknown or not MoE.
    LLAMA_API int32_t llama_blurry_sharp_n_experts_used(
            const struct llama_blurry_sharp_context * bsctx);

    // Check whether a tensor name corresponds to a merged expert tensor
    // (ffn_gate_exps, ffn_up_exps, ffn_down_exps).
    // Returns true if the tensor is a merged expert tensor.
    LLAMA_API bool llama_blurry_sharp_is_expert_tensor(
            const char * tensor_name);

    // -----------------------------------------------------------------------
    // MoE Router Recording
    //
    // During llama_decode(), MoE layers select top-k experts per token via a
    // learned router (gate).  These functions allow recording those selections
    // so that external code (e.g. the blurry-sharp overlay) can later sharpen
    // only the exact experts that were active during a draft phase.
    //
    // Usage:
    //   1. llama_router_start_recording(ctx)   — clears old data, starts recording
    //   2. llama_decode(ctx, batch) [one or more times]
    //   3. llama_router_get_layer_experts(ctx, layer, ...) — query results
    //   4. llama_router_stop_recording(ctx)    — stop (data persists until clear/start)
    //
    // Recording works by intercepting the "ffn_moe_topk-{layer}" tensors
    // during graph execution via the backend scheduler eval callback.
    // The overhead is minimal: only a few bytes per MoE layer per token
    // are copied from the compute tensor to a persistent set.
    // -----------------------------------------------------------------------

    // Start recording MoE router expert selections.
    // Clears any previously recorded data.
    // Recording persists across multiple llama_decode() calls until stopped.
    LLAMA_API void llama_router_start_recording(struct llama_context * ctx);

    // Stop recording.  Recorded data persists until cleared or next start.
    LLAMA_API void llama_router_stop_recording(struct llama_context * ctx);

    // Check if router recording is currently active.
    LLAMA_API bool llama_router_is_recording(const struct llama_context * ctx);

    // Get the unique expert IDs used in a specific model layer across all
    // recorded llama_decode() calls since the last start/clear.
    //
    // out_expert_ids: buffer to receive unique expert IDs (sorted ascending).
    //                 Pass NULL to just query the count.
    // max_ids:        capacity of out_expert_ids (ignored when NULL).
    //
    // Returns the number of unique expert IDs for that layer,
    //         0 if the layer had no MoE routing recorded,
    //         or negative on error.
    LLAMA_API int32_t llama_router_get_layer_experts(
            const struct llama_context * ctx,
            int32_t                      layer_idx,
            int32_t                    * out_expert_ids,
            int32_t                      max_ids);

    // Get the number of distinct MoE layers that have recorded data.
    LLAMA_API int32_t llama_router_n_recorded_layers(const struct llama_context * ctx);

    // Clear all recorded router data without stopping recording.
    LLAMA_API void llama_router_clear(struct llama_context * ctx);

    // Get the number of recorded token positions for a layer.
    // Returns 0 if no per-token data exists for this layer.
    LLAMA_API int32_t llama_router_n_tokens_for_layer(
            const struct llama_context * ctx,
            int32_t                      layer_idx);

    // Get the unique expert IDs used across a range of token positions
    // [token_start, token_end) for a specific layer.  This computes the
    // union over only the requested token range rather than ALL tokens,
    // allowing callers to sharpen fewer experts when verifying a subset
    // of draft positions.
    //
    // out_expert_ids: buffer to receive unique expert IDs (sorted ascending).
    //                 Pass NULL to just query the count.
    // max_ids:        capacity of out_expert_ids (ignored when NULL).
    //
    // Returns the number of unique expert IDs in the requested range,
    //         0 if no data exists, or negative on error.
    LLAMA_API int32_t llama_router_get_token_range_experts(
            const struct llama_context * ctx,
            int32_t                      layer_idx,
            int32_t                      token_start,
            int32_t                      token_end,
            int32_t                    * out_expert_ids,
            int32_t                      max_ids);

    // -----------------------------------------------------------------------
    // JIT (Just-In-Time) Per-Layer Sharpening
    //
    // Instead of sharpening ALL layers at once (which requires the entire
    // sharp model to be resident in memory simultaneously), JIT sharpening
    // hooks into the forward pass via the eval callback and sharpens ONE
    // layer at a time:
    //
    //   1. The MoE gate/router runs with blurry weights → selects experts
    //   2. The eval callback intercepts "ffn_moe_topk-{il}" → reads expert IDs
    //   3. Sharpens ONLY that layer's expert weight tensors (zero-copy mmap
    //      on CPU, in-place overwrite on GPU for same-type)
    //   4. The expert FFN ops run with sharp weights
    //   5. When the next layer's topk fires, the previous layer is restored
    //
    // Memory usage: bounded to ~one layer's worth of sharp data at a time,
    // regardless of total model size.  No re-decode needed — sharp weights
    // are used during the original forward pass.
    //
    // Usage:
    //   llama_blurry_sharp_start_jit(bsctx, ctx);
    //   llama_decode(ctx, batch);
    //   llama_blurry_sharp_stop_jit(bsctx, ctx);
    //
    // Requires single-slot mode (-np 1).  Automatically enables router
    // recording so expert selections are available after decode.
    // -----------------------------------------------------------------------

    // Start per-layer JIT sharpening for subsequent llama_decode() calls.
    // Installs an eval callback that sharpens/restores one layer at a time.
    // When host_only is true, only CPU-resident MoE layers are sharpened;
    // GPU layers are left at blurry quality to preserve CUDA graph capture.
    LLAMA_API void llama_blurry_sharp_start_jit(
            struct llama_blurry_sharp_context * bsctx,
            struct llama_context              * ctx,
            bool                                host_only);

    // Stop JIT sharpening.  Restores any still-sharpened layer and removes
    // the eval callback.  Must be called after llama_decode() returns.
    LLAMA_API void llama_blurry_sharp_stop_jit(
            struct llama_blurry_sharp_context * bsctx,
            struct llama_context              * ctx);

    // Set which layers the JIT callback should sharpen.
    // When set, layers NOT in this list are skipped (blurry weights used).
    // Pass n_layers=0 to clear the filter (sharpen all layers).
    //
    // Use this to limit JIT sharpening to the most impactful layers
    // (e.g., first half + last half) instead of all ~90 MoE layers.
    // Dramatically reduces I/O: 8 layers × ~98 MB vs 90 layers × ~98 MB.
    LLAMA_API void llama_blurry_sharp_set_jit_layers(
            struct llama_context * ctx,
            const int32_t        * layer_indices,
            int32_t                n_layers);

    //
    // MTP
    //
    //

    LLAMA_API int32_t llama_model_n_nextn_layer(const struct llama_model * model);

    // Set which, if any, MTP operation the context will use
    LLAMA_API void llama_set_mtp_op_type(struct llama_context * ctx, enum llama_mtp_op_type mtp_op_type);

    LLAMA_API void llama_set_draft_input_hidden_state(struct llama_context * ctx, const float * hidden_state);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_API_INTERNAL

#include <random>
#include <string>
#include <vector>

struct ggml_tensor;

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
);


// Randomly selects a token from the candidates based on their probabilities using given std::mt19937.
// This is a temporary workaround in order to fix race conditions when sampling with multiple sequences.
llama_token llama_sample_token_with_rng(struct llama_context * ctx, llama_token_data_array * candidates, std::mt19937 & rng);

#endif // LLAMA_API_INTERNAL

#endif // LLAMA_H
