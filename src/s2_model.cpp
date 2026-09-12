#include "../include/s2_model.h"
#if defined(GGML_USE_CUDA)
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_METAL)
#  include "ggml-metal.h"
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <limits>
#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace s2 {

static std::FILE * open_binary_input_utf8(const std::string & path) {
    // GGML's public fopen wrapper treats paths as UTF-8 and uses _wfopen on Windows.
    return path.empty() ? nullptr : ggml_fopen(path.c_str(), "rb");
}

#if defined(GGML_USE_CUDA)
static bool cuda_problem_embedding_type(enum ggml_type type) {
    return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}
#endif

static bool file_size_u64(std::FILE * f, uint64_t & out) {
    if (!f) return false;
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    const __int64 pos = _ftelli64(f);
    if (pos < 0) return false;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t pos = ftello(f);
    if (pos < 0) return false;
#endif
    out = static_cast<uint64_t>(pos);
    return true;
}

// ---------------------------------------------------------------------------
// Helpers (graph-level, no side effects)
// ---------------------------------------------------------------------------

static ggml_tensor * repeat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                    const char * label = "repeat") {
    if (!ctx || !a || !b || !ggml_can_repeat(a, b)) {
        throw std::runtime_error(std::string(label) + ": incompatible repeat shapes");
    }
    return ggml_repeat(ctx, a, b);
}

static bool same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) return false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

static ggml_tensor * add_same_shape_checked(ggml_context * ctx, ggml_tensor * a,
                                             ggml_tensor * b, const char * label) {
    if (!ctx || !same_shape(a, b)) {
        throw std::runtime_error(std::string(label) + ": incompatible add shapes");
    }
    return ggml_add(ctx, a, b);
}

static ggml_tensor * cpy_same_shape_checked(ggml_context * ctx, ggml_tensor * src,
                                             ggml_tensor * dst, const char * label) {
    if (!ctx || !same_shape(src, dst)) {
        throw std::runtime_error(std::string(label) + ": incompatible copy shapes");
    }
    return ggml_cpy(ctx, src, dst);
}

static ggml_tensor * swiglu_split_checked(ggml_context * ctx, ggml_tensor * gate,
                                           ggml_tensor * up, const char * label) {
    if (!ctx || !same_shape(gate, up)) {
        throw std::runtime_error(std::string(label) + ": incompatible gate/up shapes");
    }
    return ggml_swiglu_split(ctx, gate, up);
}

static bool active_backend_is_metal(ggml_backend_t backend) {
#if defined(GGML_USE_METAL)
    return backend != nullptr && ggml_backend_is_metal(backend);
#else
    (void) backend;
    return false;
#endif
}

struct GgmlContextGuard {
    ggml_context * ptr = nullptr;
    explicit GgmlContextGuard(ggml_context * p) : ptr(p) {}
    ~GgmlContextGuard() { if (ptr) ggml_free(ptr); }
    GgmlContextGuard(const GgmlContextGuard &) = delete;
    GgmlContextGuard & operator=(const GgmlContextGuard &) = delete;
};

class GallocrSwapGuard {
public:
    GallocrSwapGuard(ggml_gallocr_t & slot, ggml_gallocr_t replacement)
        : slot_(slot), displaced_(replacement) {
        std::swap(slot_, displaced_);
    }
    ~GallocrSwapGuard() {
        std::swap(slot_, displaced_);
        if (displaced_) ggml_gallocr_free(displaced_);
    }
    GallocrSwapGuard(const GallocrSwapGuard &) = delete;
    GallocrSwapGuard & operator=(const GallocrSwapGuard &) = delete;
private:
    ggml_gallocr_t & slot_;
    ggml_gallocr_t displaced_ = nullptr;
};

static ggml_tensor * mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                     const char * label = "mul_mat") {
    if (!ctx || !a || !b || a->ne[2] <= 0 || a->ne[3] <= 0) {
        throw std::runtime_error(std::string(label) + ": null/invalid matrix tensor");
    }
    const bool can_mul =
        a->ne[0] == b->ne[0] &&
        (b->ne[2] % a->ne[2] == 0) &&
        (b->ne[3] % a->ne[3] == 0);
    if (!can_mul || ggml_is_transposed(a)) {
        throw std::runtime_error(std::string(label) + ": incompatible matrix shapes");
    }
    return ggml_mul_mat(ctx, a, b);
}

static ggml_tensor * rms_norm_weighted(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * weight, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * w = weight;
    if (w->type != cur->type) {
        w = ggml_cast(ctx, w, cur->type);
    }
    w = repeat_checked(ctx, w, cur, "repeat:rms_norm");
    return ggml_mul(ctx, cur, w);
}

static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x,
                                              int32_t repeat_factor) {
    if (!ctx || !x || repeat_factor <= 0)
        throw std::runtime_error("repeat_interleave_heads: invalid input");
    if (repeat_factor == 1) return x;
    if (x->ne[1] > std::numeric_limits<int64_t>::max() / repeat_factor)
        throw std::runtime_error("repeat_interleave_heads: shape overflow");
    ggml_tensor * xf = (x->type != GGML_TYPE_F32) ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, xf),
                                        xf->ne[0], 1, xf->ne[1], xf->ne[2]);
    ggml_tensor * target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               xf->ne[0], repeat_factor, xf->ne[1], xf->ne[2]);
    ggml_tensor * repeated = repeat_checked(ctx, x4, target, "repeat:interleave_heads");
    return ggml_reshape_3d(ctx, ggml_cont(ctx, repeated),
                           xf->ne[0], xf->ne[1] * repeat_factor, xf->ne[2]);
}

static ggml_tensor * last_token_view(ggml_context * ctx, ggml_tensor * x, int32_t n_tokens) {
    if (!ctx || !x || n_tokens <= 0 || static_cast<int64_t>(n_tokens) > x->ne[1])
        throw std::runtime_error("last_token_view: invalid token index");
    const uint64_t index = static_cast<uint64_t>(n_tokens - 1);
    if (x->nb[1] != 0 && index > std::numeric_limits<size_t>::max() / x->nb[1])
        throw std::runtime_error("last_token_view: byte offset overflow");
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], static_cast<size_t>(index) * x->nb[1]);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SlowARModel::SlowARModel() {}

SlowARModel::~SlowARModel() {
    unload();
}

void SlowARModel::unload() {
    free_kv_cache();
    if (weights_.model_buf) ggml_backend_buffer_free(weights_.model_buf);
    if (weights_.ctx_w)     ggml_free(weights_.ctx_w);
    if (fast_allocr_)       ggml_gallocr_free(fast_allocr_);
    if (allocr_)            ggml_gallocr_free(allocr_);
    if (backend_ && backend_ != backend_cpu_) ggml_backend_free(backend_);
    if (backend_cpu_)       ggml_backend_free(backend_cpu_);

    hparams_       = ModelHParams{};
    weights_       = ModelWeights{};
    backend_       = nullptr;
    backend_cpu_   = nullptr;
    cuda_mode_     = false;
    cuda_host_embeddings_ = false;
    cuda_host_codebook_embeddings_ = false;
    cuda_fast_host_embeddings_ = false;
    host_embeddings_.clear();
    host_codebook_embeddings_.clear();
    host_fast_embeddings_.clear();
    allocr_        = nullptr;
    fast_allocr_   = nullptr;
    ctx_kv_        = nullptr;
    kv_buf_        = nullptr;
    memory_k_      = nullptr;
    memory_v_      = nullptr;
    max_seq_len_   = 0;
    n_past_        = 0;
    std::vector<uint8_t>().swap(graph_ctx_buf_);
    std::vector<uint8_t>().swap(fast_graph_ctx_buf_);
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool SlowARModel::load(const std::string & gguf_path, int32_t vulkan_device) {
    unload();
    // GPU backend init: #if at the outer level so MSVC/CUDA does not mis-track
    // function scope when preprocessor branches are nested inside C++ braces.
#if defined(GGML_USE_CUDA)
    if (vulkan_device >= 0) {
        backend_ = ggml_backend_cuda_init(vulkan_device);
        if (!backend_) {
            std::cerr << "[Model] CUDA init failed on device " << vulkan_device
                      << ", falling back to CPU." << std::endl;
        } else {
            std::cout << "[Model] CUDA backend on device " << vulkan_device << std::endl;
            cuda_mode_ = true;
        }
    }
#elif defined(GGML_USE_VULKAN)
    if (vulkan_device >= 0) {
        backend_ = ggml_backend_vk_init(static_cast<size_t>(vulkan_device));
        if (!backend_) {
            std::cerr << "[Model] Vulkan init failed, falling back to CPU." << std::endl;
        }
    }
#elif defined(GGML_USE_METAL)
    if (vulkan_device >= 0) {
        backend_ = ggml_backend_metal_init();
        if (!backend_) std::cerr << "[Model] Metal init failed, falling back to CPU." << std::endl;
        else std::cout << "[Model] Metal backend" << std::endl;
    }
#else
    if (vulkan_device >= 0) {
        std::cerr << "[Model] No GPU backend compiled, falling back to CPU." << std::endl;
    }
#endif
    // The CUDA embedding workaround below performs host row dequantization
    // directly from retained GGUF bytes, so a second GGML CPU backend is not
    // needed when a GPU backend initialized successfully. Create CPU only as
    // the actual model fallback/backend.
    if (!backend_) {
        backend_cpu_ = ggml_backend_cpu_init();
        if (!backend_cpu_) {
            std::cerr << "[Model] Failed to init CPU backend." << std::endl;
            unload();
            return false;
        }
        backend_ = backend_cpu_;
    }
    if (!backend_) {
        std::cerr << "[Model] Failed to init any GGML backend." << std::endl;
        unload();
        return false;
    }

    allocr_      = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    fast_allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocr_ || !fast_allocr_) {
        std::cerr << "[Model] Failed to create graph allocators." << std::endl;
        unload();
        return false;
    }

    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&weights_.ctx_w };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[Model] Failed to load GGUF from " << gguf_path << std::endl;
        unload();
        return false;
    }

    std::cout << "[Model] Reading metadata from " << gguf_path << std::endl;

    // Helpers to read GGUF metadata. gguf_get_val_* asserts if the stored
    // type does not match the getter, so validate the type first and reject a
    // malformed/incompatible file cleanly instead of aborting the process.
    bool metadata_types_ok = true;
    auto bad_type = [&](const char * key, enum gguf_type expected, enum gguf_type actual) {
        std::cerr << "[GGUF] wrong type for " << key << ": expected "
                  << gguf_type_name(expected) << ", got " << gguf_type_name(actual) << "\n";
        metadata_types_ok = false;
    };
    auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
        int64_t id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; return def; }
        const enum gguf_type type = gguf_get_kv_type(ctx_gguf, id);
        if (type != GGUF_TYPE_UINT32) { bad_type(key, GGUF_TYPE_UINT32, type); return def; }
        uint32_t v = gguf_get_val_u32(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << v << "\n";
        return v;
    };
    auto get_i32_from_u32 = [&](const char * key, uint32_t def) -> int32_t {
        const uint32_t v = get_u32(key, def);
        if (v > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            std::cerr << "[GGUF] out-of-range UINT32 for signed model field "
                      << key << ": " << v << "\n";
            metadata_types_ok = false;
            return 0;
        }
        return static_cast<int32_t>(v);
    };
    auto get_f32 = [&](const char * key, float def) -> float {
        int64_t id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; return def; }
        const enum gguf_type type = gguf_get_kv_type(ctx_gguf, id);
        if (type != GGUF_TYPE_FLOAT32) { bad_type(key, GGUF_TYPE_FLOAT32, type); return def; }
        float v = gguf_get_val_f32(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << v << "\n";
        return v;
    };
    auto get_bool = [&](const char * key, bool def) -> bool {
        int64_t id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << (def?"true":"false") << ")\n"; return def; }
        const enum gguf_type type = gguf_get_kv_type(ctx_gguf, id);
        if (type != GGUF_TYPE_BOOL) { bad_type(key, GGUF_TYPE_BOOL, type); return def; }
        bool v = gguf_get_val_bool(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << (v?"true":"false") << "\n";
        return v;
    };

    hparams_ = ModelHParams();

    // Determine architecture prefix from the file
    std::string arch_prefix = "fish-speech.";
    {
        int64_t arch_id = gguf_find_key(ctx_gguf, "general.architecture");
        if (arch_id >= 0) {
            const enum gguf_type type = gguf_get_kv_type(ctx_gguf, arch_id);
            if (type != GGUF_TYPE_STRING) {
                bad_type("general.architecture", GGUF_TYPE_STRING, type);
            } else {
                std::string arch = gguf_get_val_str(ctx_gguf, arch_id);
                arch_prefix = arch + ".";
                hparams_.has_fast_decoder = (arch == "fish-speech");
                std::cout << "[Model] Architecture: " << arch << std::endl;
            }
        }
    }

    // Main model hparams (from arch-prefixed keys)
    hparams_.context_length      = get_i32_from_u32((arch_prefix + "context_length").c_str(), 32768);
    hparams_.vocab_size          = get_i32_from_u32((arch_prefix + "vocab_size").c_str(), 155776);
    hparams_.embedding_length    = get_i32_from_u32((arch_prefix + "embedding_length").c_str(), 2560);
    hparams_.feed_forward_length = get_i32_from_u32((arch_prefix + "feed_forward_length").c_str(), 9728);
    hparams_.block_count         = get_i32_from_u32((arch_prefix + "block_count").c_str(), 36);
    hparams_.head_count          = get_i32_from_u32((arch_prefix + "attention.head_count").c_str(), 32);
    hparams_.head_count_kv       = get_i32_from_u32((arch_prefix + "attention.head_count_kv").c_str(), 8);
    hparams_.rope_freq_base      = get_f32((arch_prefix + "rope.freq_base").c_str(), 1e6f);
    hparams_.rms_norm_eps        = get_f32((arch_prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

    // Fish-speech specific keys
    hparams_.codebook_size            = get_i32_from_u32("fish_speech.codebook_size", 4096);
    hparams_.num_codebooks            = get_i32_from_u32("fish_speech.num_codebooks", 10);
    hparams_.semantic_begin_id        = get_i32_from_u32("fish_speech.semantic_begin_id", 151678);
    hparams_.semantic_end_id          = get_i32_from_u32("fish_speech.semantic_end_id", 155773);
    hparams_.tie_word_embeddings      = get_bool("fish_speech.tie_word_embeddings", true);
    hparams_.attention_qk_norm        = get_bool("fish_speech.attention_qk_norm", false);
    hparams_.scale_codebook_embeddings = get_bool("fish_speech.scale_codebook_embeddings", false);

    // Fast decoder hparams
    if (hparams_.has_fast_decoder) {
        hparams_.fast_context_length   = get_i32_from_u32("fish_speech.fast_context_length", 11);
        hparams_.fast_embedding_length = get_i32_from_u32("fish_speech.fast_embedding_length", 2560);
        hparams_.fast_feed_forward_length = get_i32_from_u32("fish_speech.fast_feed_forward_length", 9728);
        hparams_.fast_block_count      = get_i32_from_u32("fish_speech.fast_block_count", 4);
        hparams_.fast_head_count       = get_i32_from_u32("fish_speech.fast_head_count", 32);
        hparams_.fast_head_count_kv    = get_i32_from_u32("fish_speech.fast_head_count_kv", 8);
        hparams_.fast_head_dim         = get_i32_from_u32("fish_speech.fast_head_dim", 128);
        hparams_.fast_rope_freq_base   = get_f32("fish_speech.fast_rope_freq_base", 1e6f);
        hparams_.fast_rms_norm_eps     = get_f32("fish_speech.fast_layer_norm_rms_eps", 1e-6f);
        hparams_.fast_attention_qk_norm = get_bool("fish_speech.fast_attention_qk_norm", false);
        hparams_.fast_has_project_in   = get_bool("fish_speech.fast_project_in", false);
    }

    // GGUF is input data. Reject impossible/hostile dimensions before any
    // vector resize or graph allocation (uint32 -> int32 casts can wrap).
    auto sane_pos = [](int32_t v, int32_t max) { return v > 0 && v <= max; };
    const bool main_hparams_ok =
        sane_pos(hparams_.context_length, 1000000) &&
        sane_pos(hparams_.vocab_size, 10000000) &&
        sane_pos(hparams_.embedding_length, 1000000) &&
        sane_pos(hparams_.feed_forward_length, 10000000) &&
        sane_pos(hparams_.block_count, 4096) &&
        sane_pos(hparams_.head_count, 4096) &&
        sane_pos(hparams_.head_count_kv, hparams_.head_count) &&
        (hparams_.head_count % hparams_.head_count_kv == 0) &&
        sane_pos(hparams_.codebook_size, 1000000) &&
        sane_pos(hparams_.num_codebooks, 1024) &&
        hparams_.semantic_begin_id >= 0 &&
        hparams_.semantic_end_id >= hparams_.semantic_begin_id &&
        hparams_.semantic_end_id < hparams_.vocab_size &&
        (static_cast<int64_t>(hparams_.semantic_end_id) - hparams_.semantic_begin_id + 1 ==
         static_cast<int64_t>(hparams_.codebook_size)) &&
        std::isfinite(hparams_.rope_freq_base) && hparams_.rope_freq_base > 0.0f &&
        std::isfinite(hparams_.rms_norm_eps) && hparams_.rms_norm_eps > 0.0f;
    bool fast_hparams_ok = true;
    if (hparams_.has_fast_decoder) {
        fast_hparams_ok =
            sane_pos(hparams_.fast_context_length, 1000000) &&
            sane_pos(hparams_.fast_embedding_length, 1000000) &&
            sane_pos(hparams_.fast_feed_forward_length, 10000000) &&
            sane_pos(hparams_.fast_block_count, 4096) &&
            sane_pos(hparams_.fast_head_count, 4096) &&
            sane_pos(hparams_.fast_head_count_kv, hparams_.fast_head_count) &&
            (hparams_.fast_head_count % hparams_.fast_head_count_kv == 0) &&
            sane_pos(hparams_.fast_head_dim, 1000000) &&
            hparams_.fast_context_length >= hparams_.num_codebooks &&
            std::isfinite(hparams_.fast_rope_freq_base) && hparams_.fast_rope_freq_base > 0.0f &&
            std::isfinite(hparams_.fast_rms_norm_eps) && hparams_.fast_rms_norm_eps > 0.0f;
    }
    if (!metadata_types_ok || !main_hparams_ok || !fast_hparams_ok) {
        std::cerr << "[Model] Invalid/out-of-range GGUF metadata." << std::endl;
        gguf_free(ctx_gguf);
        unload();
        return false;
    }

    std::cout << "[Model] Layers: " << hparams_.block_count
              << ", Dim: " << hparams_.embedding_length
              << ", Vocab: " << hparams_.vocab_size
              << ", head_count: " << hparams_.head_count
              << ", has_fast_decoder: " << hparams_.has_fast_decoder << std::endl;

    // ---------------------------------------------------------------------------
    // Load tensor pointers (metadata only -- data loaded below)
    // ---------------------------------------------------------------------------
    auto req_t = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, name.c_str());
        if (!t) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return t;
    };

    try {
        weights_.embeddings          = req_t("embeddings.weight");
        weights_.output              = hparams_.tie_word_embeddings ? weights_.embeddings : req_t("output.weight");
        weights_.codebook_embeddings = req_t("codebook_embeddings.weight");
        weights_.norm                = req_t("norm.weight");

        weights_.layers.resize(hparams_.block_count);
        for (int32_t i = 0; i < hparams_.block_count; ++i) {
            auto & layer = weights_.layers[i];
            std::string stem = "layers." + std::to_string(i) + ".";

            layer.attention_norm = req_t(stem + "attention_norm.weight");
            layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
            layer.wqkv           = req_t(stem + "attention.wqkv.weight");
            layer.wo             = req_t(stem + "attention.wo.weight");
            layer.w1             = req_t(stem + "feed_forward.w1.weight");
            layer.w2             = req_t(stem + "feed_forward.w2.weight");
            layer.w3             = req_t(stem + "feed_forward.w3.weight");

            if (hparams_.attention_qk_norm) {
                layer.q_norm = req_t(stem + "attention.q_norm.weight");
                layer.k_norm = req_t(stem + "attention.k_norm.weight");
            }
        }

        if (hparams_.has_fast_decoder) {
            if (hparams_.fast_has_project_in) {
                weights_.fast_project_in = req_t("fast_project_in.weight");
            }
            weights_.fast_embeddings = req_t("fast_embeddings.weight");
            weights_.fast_norm       = req_t("fast_norm.weight");
            weights_.fast_output     = req_t("fast_output.weight");

            weights_.fast_layers.resize(hparams_.fast_block_count);
            for (int32_t i = 0; i < hparams_.fast_block_count; ++i) {
                auto & layer = weights_.fast_layers[i];
                std::string stem = "fast_layers." + std::to_string(i) + ".";

                layer.attention_norm = req_t(stem + "attention_norm.weight");
                layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
                layer.wqkv           = req_t(stem + "attention.wqkv.weight");
                layer.wo             = req_t(stem + "attention.wo.weight");
                layer.w1             = req_t(stem + "feed_forward.w1.weight");
                layer.w2             = req_t(stem + "feed_forward.w2.weight");
                layer.w3             = req_t(stem + "feed_forward.w3.weight");

                if (hparams_.fast_attention_qk_norm) {
                    layer.q_norm = req_t(stem + "attention.q_norm.weight");
                    layer.k_norm = req_t(stem + "attention.k_norm.weight");
                }
            }
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] " << e.what() << std::endl;
        gguf_free(ctx_gguf);
        unload();
        return false;
    }

    // Validate tensor shapes before any GGML graph is built. Malformed or
    // incompatible GGUFs should fail with a useful error instead of reaching
    // ggml_view/mul_mat/repeat assertions in a backend-specific code path.
    auto vec_is = [](const ggml_tensor * t, int64_t n) {
        return t && ggml_nelements(t) == n;
    };
    auto mat_is = [](const ggml_tensor * t, int64_t n0, int64_t n1_min, bool exact_n1 = true) {
        if (!t || t->ne[0] != n0 || t->ne[1] <= 0 || t->ne[2] != 1 || t->ne[3] != 1) return false;
        return exact_n1 ? t->ne[1] == n1_min : t->ne[1] >= n1_min;
    };
    auto bad_shape = [](const std::string & n, const ggml_tensor * t) {
        std::cerr << "[Model] Invalid tensor shape: " << n;
        if (t) std::cerr << " = (" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << ")";
        std::cerr << "\n";
    };

    bool shapes_ok = true;
    const int64_t dim = hparams_.embedding_length;
    int64_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty()) {
        head_dim = weights_.layers[0].q_norm ? weights_.layers[0].q_norm->ne[0] : 0;
    } else if (!weights_.layers.empty() && hparams_.head_count > 0 &&
               weights_.layers[0].wo->ne[0] % hparams_.head_count == 0) {
        head_dim = weights_.layers[0].wo->ne[0] / hparams_.head_count;
    }
    const int64_t q_size64  = static_cast<int64_t>(hparams_.head_count) * head_dim;
    const int64_t kv_size64 = static_cast<int64_t>(hparams_.head_count_kv) * head_dim;
    if (head_dim <= 0 || q_size64 <= 0 || kv_size64 <= 0 ||
        q_size64 > std::numeric_limits<int32_t>::max() ||
        kv_size64 > std::numeric_limits<int32_t>::max()) {
        std::cerr << "[Model] Invalid attention head dimensions.\n";
        shapes_ok = false;
    }

    auto check = [&](bool ok, const std::string & n, ggml_tensor * t) {
        if (!ok) { bad_shape(n, t); shapes_ok = false; }
    };
    check(mat_is(weights_.embeddings, dim, hparams_.vocab_size, false), "embeddings.weight", weights_.embeddings);
    check(mat_is(weights_.output, dim, hparams_.vocab_size, false),
          hparams_.tie_word_embeddings ? "embeddings.weight(output)" : "output.weight", weights_.output);
    const int64_t cb_rows = static_cast<int64_t>(hparams_.num_codebooks) * hparams_.codebook_size;
    check(cb_rows > 0 && mat_is(weights_.codebook_embeddings, dim, cb_rows, false),
          "codebook_embeddings.weight", weights_.codebook_embeddings);
    check(vec_is(weights_.norm, dim), "norm.weight", weights_.norm);

    if (shapes_ok) {
        for (int32_t i = 0; i < hparams_.block_count; ++i) {
            auto & l = weights_.layers[i];
            const std::string stem = "layers." + std::to_string(i) + ".";
            check(vec_is(l.attention_norm, dim), stem + "attention_norm.weight", l.attention_norm);
            check(vec_is(l.ffn_norm, dim), stem + "ffn_norm.weight", l.ffn_norm);
            check(mat_is(l.wqkv, dim, q_size64 + 2 * kv_size64), stem + "attention.wqkv.weight", l.wqkv);
            check(mat_is(l.wo, q_size64, dim), stem + "attention.wo.weight", l.wo);
            check(mat_is(l.w1, dim, hparams_.feed_forward_length), stem + "feed_forward.w1.weight", l.w1);
            check(mat_is(l.w3, dim, hparams_.feed_forward_length), stem + "feed_forward.w3.weight", l.w3);
            check(mat_is(l.w2, hparams_.feed_forward_length, dim), stem + "feed_forward.w2.weight", l.w2);
            if (hparams_.attention_qk_norm) {
                check(vec_is(l.q_norm, head_dim), stem + "attention.q_norm.weight", l.q_norm);
                check(vec_is(l.k_norm, head_dim), stem + "attention.k_norm.weight", l.k_norm);
            }
        }
    }

    if (hparams_.has_fast_decoder && shapes_ok) {
        const int64_t fdim = hparams_.fast_embedding_length;
        const int64_t fhead = hparams_.fast_head_dim > 0
            ? hparams_.fast_head_dim
            : (hparams_.fast_head_count > 0 && fdim % hparams_.fast_head_count == 0
               ? fdim / hparams_.fast_head_count : 0);
        const int64_t fq = static_cast<int64_t>(hparams_.fast_head_count) * fhead;
        const int64_t fkv = static_cast<int64_t>(hparams_.fast_head_count_kv) * fhead;
        if (fhead <= 0 || fq <= 0 || fkv <= 0 || fq > std::numeric_limits<int32_t>::max() ||
            fkv > std::numeric_limits<int32_t>::max()) {
            std::cerr << "[Model] Invalid fast-decoder head dimensions.\n";
            shapes_ok = false;
        }
        if (weights_.fast_project_in)
            check(mat_is(weights_.fast_project_in, dim, fdim), "fast_project_in.weight", weights_.fast_project_in);
        else if (dim != fdim) {
            std::cerr << "[Model] Fast decoder requires project_in when slow/fast embedding dimensions differ.\n";
            shapes_ok = false;
        }
        check(mat_is(weights_.fast_embeddings, fdim, hparams_.codebook_size, false), "fast_embeddings.weight", weights_.fast_embeddings);
        check(vec_is(weights_.fast_norm, fdim), "fast_norm.weight", weights_.fast_norm);
        check(mat_is(weights_.fast_output, fdim, hparams_.codebook_size, false), "fast_output.weight", weights_.fast_output);
        if (shapes_ok) {
            for (int32_t i = 0; i < hparams_.fast_block_count; ++i) {
                auto & l = weights_.fast_layers[i];
                const std::string stem = "fast_layers." + std::to_string(i) + ".";
                check(vec_is(l.attention_norm, fdim), stem + "attention_norm.weight", l.attention_norm);
                check(vec_is(l.ffn_norm, fdim), stem + "ffn_norm.weight", l.ffn_norm);
                check(mat_is(l.wqkv, fdim, fq + 2 * fkv), stem + "attention.wqkv.weight", l.wqkv);
                check(mat_is(l.wo, fq, fdim), stem + "attention.wo.weight", l.wo);
                check(mat_is(l.w1, fdim, hparams_.fast_feed_forward_length), stem + "feed_forward.w1.weight", l.w1);
                check(mat_is(l.w3, fdim, hparams_.fast_feed_forward_length), stem + "feed_forward.w3.weight", l.w3);
                check(mat_is(l.w2, hparams_.fast_feed_forward_length, fdim), stem + "feed_forward.w2.weight", l.w2);
                if (hparams_.fast_attention_qk_norm) {
                    check(vec_is(l.q_norm, fhead), stem + "attention.q_norm.weight", l.q_norm);
                    check(vec_is(l.k_norm, fhead), stem + "attention.k_norm.weight", l.k_norm);
                }
            }
        }
    }
    if (!shapes_ok) {
        gguf_free(ctx_gguf);
        unload();
        return false;
    }

#if defined(GGML_USE_CUDA)
    // K-quant get_rows has historically been problematic on CUDA, including
    // Q2_K/Q3_K and voice-prefill paths.  Keep the tensors in their original
    // quantized layout and retain a host-side copy only for the embedding tables
    // that need the workaround.  During inference we dequantize just the rows
    // referenced by the current batch and upload the resulting F32 activations.
    // This avoids both CUDA get_rows and V5's full-table F32 + F16 RAM spike.
    cuda_host_embeddings_ = cuda_mode_ &&
        cuda_problem_embedding_type(weights_.embeddings->type);
    cuda_host_codebook_embeddings_ = cuda_mode_ &&
        cuda_problem_embedding_type(weights_.codebook_embeddings->type);
    cuda_fast_host_embeddings_ = cuda_mode_ && weights_.fast_embeddings &&
        cuda_problem_embedding_type(weights_.fast_embeddings->type);
#endif

    // Validate the physical file before allocating the potentially very large
    // backend weight buffer. A truncated/hostile GGUF must fail cheaply rather
    // than first consuming RAM/VRAM and only then discovering missing bytes.
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    const int64_t n_tensors  = gguf_get_n_tensors(ctx_gguf);
    std::FILE * f = open_binary_input_utf8(gguf_path);
    if (!f) {
        std::cerr << "[Model] Cannot reopen " << gguf_path << " for data loading." << std::endl;
        gguf_free(ctx_gguf);
        unload();
        return false;
    }
    uint64_t physical_file_size = 0;
    if (!file_size_u64(f, physical_file_size)) {
        std::cerr << "[Model] Cannot determine GGUF file size." << std::endl;
        std::fclose(f); gguf_free(ctx_gguf); unload(); return false;
    }

    try {
        for (int64_t ti = 0; ti < n_tensors; ++ti) {
            const char * tname = gguf_get_tensor_name(ctx_gguf, ti);
            ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, tname);
            if (!t) continue;
            const size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, ti);
            if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset)
                throw std::runtime_error(std::string("invalid tensor offset: ") + tname);
            const size_t toff = data_offset + tensor_offset;
            const size_t file_bytes = gguf_get_tensor_size(ctx_gguf, ti);
            const uint64_t toff64 = static_cast<uint64_t>(toff);
            if (toff64 > physical_file_size ||
                static_cast<uint64_t>(file_bytes) > physical_file_size - toff64)
                throw std::runtime_error(std::string("tensor extends past end of GGUF file: ") + tname);
            const enum ggml_type file_type = gguf_get_tensor_type(ctx_gguf, ti);
            if (file_type != t->type || file_bytes != ggml_nbytes(t))
                throw std::runtime_error(std::string("GGUF/backend tensor byte-size mismatch: ") + tname);
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] GGUF physical validation failed: " << e.what() << std::endl;
        std::fclose(f); gguf_free(ctx_gguf); unload(); return false;
    }

    weights_.model_buf = ggml_backend_alloc_ctx_tensors(weights_.ctx_w, backend_);
    if (!weights_.model_buf) {
        std::cerr << "[Model] Failed to allocate backend buffer for weights." << std::endl;
        std::fclose(f); gguf_free(ctx_gguf); unload(); return false;
    }

    std::vector<uint8_t> tmp;
    try {
        for (int64_t ti = 0; ti < n_tensors; ++ti) {
            const char * tname = gguf_get_tensor_name(ctx_gguf, ti);
            ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, tname);
            if (!t) continue;
            const size_t tensor_offset = gguf_get_tensor_offset(ctx_gguf, ti);
            if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset)
                throw std::runtime_error(std::string("invalid tensor offset: ") + tname);
            const size_t toff = data_offset + tensor_offset;
            const size_t file_size = gguf_get_tensor_size(ctx_gguf, ti);
            const uint64_t toff64 = static_cast<uint64_t>(toff);
            if (toff64 > physical_file_size ||
                static_cast<uint64_t>(file_size) > physical_file_size - toff64)
                throw std::runtime_error(std::string("tensor extends past end of GGUF file: ") + tname);

            const enum ggml_type file_type = gguf_get_tensor_type(ctx_gguf, ti);
            if (file_type != t->type || file_size != ggml_nbytes(t))
                throw std::runtime_error(std::string("GGUF/backend tensor byte-size mismatch: ") + tname);

#if defined(GGML_USE_CUDA)
            HostEmbeddingTable * host_table = nullptr;
            if (cuda_host_embeddings_ && t == weights_.embeddings) {
                host_table = &host_embeddings_;
            } else if (cuda_host_codebook_embeddings_ && t == weights_.codebook_embeddings) {
                host_table = &host_codebook_embeddings_;
            } else if (cuda_fast_host_embeddings_ && t == weights_.fast_embeddings) {
                host_table = &host_fast_embeddings_;
            }
#endif

            // Read CUDA host-lookup tables directly into their retained compressed
            // storage. Moving the reusable scratch vector would also transfer its
            // potentially much larger capacity from a previously-read tensor.
            uint8_t * tensor_bytes = nullptr;
#if defined(GGML_USE_CUDA)
            if (host_table) {
                host_table->data.resize(file_size);
                tensor_bytes = host_table->data.data();
            } else
#endif
            {
                if (tmp.size() < file_size) tmp.resize(file_size);
                tensor_bytes = tmp.data();
            }

#ifdef _WIN32
            const int seek_rc = (toff > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                                ? -1 : _fseeki64(f, static_cast<int64_t>(toff), SEEK_SET);
#else
            const int seek_rc = (toff > static_cast<size_t>(std::numeric_limits<off_t>::max()))
                                ? -1 : fseeko(f, static_cast<off_t>(toff), SEEK_SET);
#endif
            if (seek_rc != 0 || (file_size > 0 && std::fread(tensor_bytes, 1, file_size, f) != file_size))
                throw std::runtime_error(std::string("failed to read tensor: ") + tname);

            ggml_backend_tensor_set(t, tensor_bytes, 0, file_size);

#if defined(GGML_USE_CUDA)
            if (host_table) {
                const int64_t rows = ggml_nrows(t);
                if (t->ne[0] <= 0 || rows <= 0)
                    throw std::runtime_error(std::string("invalid host embedding shape: ") + tname);
                const size_t row_bytes = ggml_row_size(file_type, t->ne[0]);
                if (row_bytes == 0 || static_cast<uint64_t>(rows) >
                    static_cast<uint64_t>(std::numeric_limits<size_t>::max() / row_bytes) ||
                    row_bytes * static_cast<size_t>(rows) != file_size)
                    throw std::runtime_error(std::string("invalid host embedding row layout: ") + tname);
                host_table->type = file_type;
                host_table->width = t->ne[0];
                host_table->rows = rows;
                host_table->row_bytes = row_bytes;
                std::cout << "[Model] CUDA host embedding lookup: " << tname
                          << " (" << ggml_type_name(file_type) << ")\n";
            }
#endif
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] Weight loading failed: " << e.what() << std::endl;
        std::fclose(f);
        gguf_free(ctx_gguf);
        unload();
        return false;
    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);

    // Advise the kernel to drop the file pages from page cache -- the weights
    // are now in the backend buffer (VRAM) and we no longer need the cached
    // file data in RAM.
#ifdef __linux__
    {
        int fd = ::open(gguf_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif

    std::cout << "[Model] Weights loaded. Total tensors: " << n_tensors << std::endl;

    gguf_free(ctx_gguf);
    return true;
}

bool SlowARModel::decode_host_embedding_row(const HostEmbeddingTable & table,
                                            int64_t row, float * out) const {
    if (!table.valid() || !out || row < 0 || row >= table.rows || table.width <= 0) {
        return false;
    }
    if (static_cast<uint64_t>(row) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / table.row_bytes)) {
        return false;
    }
    const size_t offset = static_cast<size_t>(row) * table.row_bytes;
    if (offset > table.data.size() || table.row_bytes > table.data.size() - offset) {
        return false;
    }
    const void * src = table.data.data() + offset;

    if (table.type == GGML_TYPE_F32) {
        if (static_cast<uint64_t>(table.width) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) ||
            static_cast<size_t>(table.width) * sizeof(float) != table.row_bytes) {
            return false;
        }
        std::memcpy(out, src, table.row_bytes);
        return true;
    }

    const ggml_type_traits * tt = ggml_get_type_traits(table.type);
    if (!tt || !tt->to_float) {
        return false;
    }
    tt->to_float(src, out, table.width);
    return true;
}

// ---------------------------------------------------------------------------
// init_kv_cache()
// ---------------------------------------------------------------------------

bool SlowARModel::init_kv_cache(int32_t max_seq_len) {
    if (max_seq_len <= 0 || hparams_.context_length <= 0 || max_seq_len > hparams_.context_length) {
        std::cerr << "[Model] Invalid KV cache length: " << max_seq_len
                  << " (model context=" << hparams_.context_length << ")" << std::endl;
        return false;
    }
    // Reinitialization must release the old GGML context/buffer first.  Keep
    // the externally observable cache state empty until every validation and
    // allocation succeeds; hostile/custom metadata must not leave a non-zero
    // max_seq_len_ paired with null K/V storage after a failed re-init.
    free_kv_cache();

    const int32_t dim = hparams_.embedding_length;
    if (dim <= 0) return false;

    // head_dim: if attention_qk_norm, get from q_norm weight shape; else dim/head_count
    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        if (weights_.layers[0].q_norm->ne[0] <= 0 ||
            weights_.layers[0].q_norm->ne[0] > std::numeric_limits<int32_t>::max()) {
            std::cerr << "[Model] Invalid QK norm head dimension." << std::endl;
            return false;
        }
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        if (hparams_.head_count <= 0 || hparams_.embedding_length % hparams_.head_count != 0) {
            std::cerr << "[Model] Embedding dimension is not divisible by head count." << std::endl;
            return false;
        }
        head_dim = hparams_.embedding_length / hparams_.head_count;
    }

    const int32_t n_head_kv = hparams_.head_count_kv;
    const int32_t n_layer   = hparams_.block_count;
    if (head_dim <= 0 || n_head_kv <= 0 || n_layer <= 0) {
        std::cerr << "[Model] Invalid KV cache dimensions." << std::endl;
        return false;
    }

    // ggml_new_tensor_4d() assumes the element-count arithmetic is representable.
    // Validate hostile/custom metadata before entering GGML so this is a clean
    // load failure rather than an assertion/overflow inside the allocator.
    uint64_t kv_elems = static_cast<uint64_t>(head_dim);
    const uint64_t kv_dims[] = {
        static_cast<uint64_t>(n_head_kv),
        static_cast<uint64_t>(max_seq_len),
        static_cast<uint64_t>(n_layer),
    };
    for (uint64_t d : kv_dims) {
        if (d == 0 || kv_elems > std::numeric_limits<uint64_t>::max() / d) {
            std::cerr << "[Model] KV cache shape overflow." << std::endl;
            return false;
        }
        kv_elems *= d;
    }
    if (kv_elems > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        kv_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(uint16_t))) {
        std::cerr << "[Model] KV cache is too large for this process." << std::endl;
        return false;
    }

    const size_t ctx_kv_size = 2ull * ggml_tensor_overhead() + (1ull << 20);
    ggml_init_params p = {
        /*.mem_size =*/ ctx_kv_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx_kv_ = ggml_init(p);
    if (!ctx_kv_) {
        std::cerr << "[Model] Failed to init KV context." << std::endl;
        return false;
    }

    memory_k_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, n_head_kv, max_seq_len, n_layer);
    memory_v_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, n_head_kv, max_seq_len, n_layer);

    kv_buf_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend_);
    if (!kv_buf_) {
        std::cerr << "[Model] Failed to allocate KV cache buffer." << std::endl;
        free_kv_cache();
        return false;
    }

    ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
    ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));

    // Publish capacity only after the cache is fully usable.
    max_seq_len_ = max_seq_len;
    n_past_ = 0;
    return true;
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

void SlowARModel::reset() {
    n_past_ = 0;
}


// ---------------------------------------------------------------------------
// prefill() / step()
// ---------------------------------------------------------------------------

bool SlowARModel::prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                          int32_t n_threads, StepResult & result) {
    if (n_tokens <= 0) return false;

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    if (codebook_dim <= 0 ||
        static_cast<uint64_t>(n_tokens) * static_cast<uint64_t>(codebook_dim) != flat_tokens.size()) {
        return false;
    }

    // CUDA voice prefill is more stable when semantic/codebook prompt tokens are
    // evaluated individually. Keep ordinary text prefill batched, but once the
    // prompt contains multiple semantic frames, process one timestep at a time.
    int32_t chunk_tokens = n_tokens;
    if (cuda_mode_) {
        int32_t semantic_prompt_tokens = 0;
        for (int32_t t = 0; t < n_tokens; ++t) {
            const int32_t semantic = flat_tokens[static_cast<size_t>(t) * codebook_dim];
            if (semantic >= hparams_.semantic_begin_id && semantic <= hparams_.semantic_end_id) {
                if (++semantic_prompt_tokens > 1) {
                    chunk_tokens = 1;
                    break;
                }
            }
        }
    }

    // Use a temporary gallocr for prefill so the large compute buffer
    // (sized for n_tokens) is freed immediately after, not kept for steps.
    ggml_gallocr_t prefill_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!prefill_allocr) return false;
    GallocrSwapGuard allocator_guard(allocr_, prefill_allocr);

    bool ok = true;
    if (chunk_tokens >= n_tokens) {
        ok = eval_cached(flat_tokens, n_tokens, n_threads, result);
    } else {
        std::cout << "[Model] CUDA semantic prefill: " << n_tokens << " tokens one-by-one\n";
        StepResult chunk_result;
        std::vector<int32_t> chunk(static_cast<size_t>(codebook_dim));
        for (int32_t start = 0; start < n_tokens && ok; ++start) {
            const auto begin = flat_tokens.begin() + static_cast<ptrdiff_t>(start) * codebook_dim;
            std::copy(begin, begin + codebook_dim, chunk.begin());
            ok = eval_cached(chunk, 1, n_threads, chunk_result);
        }
        if (ok) result = std::move(chunk_result);
    }
    return ok;
}

bool SlowARModel::step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
                       StepResult & result) {
    return eval_cached(flat_tokens, 1, n_threads, result);
}

// ---------------------------------------------------------------------------
// eval_cached() -- main inference path with KV cache
// ---------------------------------------------------------------------------

bool SlowARModel::eval_cached(const std::vector<int32_t> & flat_tokens,
                               int32_t n_tokens, int32_t n_threads,
                               StepResult & result) {
    if (n_tokens <= 0) return false;

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    const int64_t expected_flat = static_cast<int64_t>(n_tokens) * static_cast<int64_t>(codebook_dim);
    if (expected_flat <= 0 || static_cast<uint64_t>(expected_flat) != static_cast<uint64_t>(flat_tokens.size())) {
        std::fprintf(stderr, "[eval_cached] expected %lld ints for %d tokens, got %zu\n",
            static_cast<long long>(expected_flat), n_tokens, flat_tokens.size());
        return false;
    }
    if (n_past_ + n_tokens > max_seq_len_) {
        std::fprintf(stderr, "[eval_cached] KV cache overflow (%d + %d > %d)\n",
            n_past_, n_tokens, max_seq_len_);
        return false;
    }

    // get_rows() assumes every index is valid. Treat prompt/profile data as
    // untrusted and fail before constructing the graph instead of relying on a
    // backend-specific assert or an out-of-bounds device read.
    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t semantic = flat_tokens[static_cast<size_t>(t) * codebook_dim];
        if (semantic < 0 || semantic >= hparams_.vocab_size) {
            std::fprintf(stderr, "[eval_cached] token id out of range: %d\n", semantic);
            return false;
        }
        const bool is_semantic = semantic >= hparams_.semantic_begin_id &&
                                 semantic <= hparams_.semantic_end_id;
        if (is_semantic) {
            for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
                const int32_t v = flat_tokens[static_cast<size_t>(t) * codebook_dim + cb + 1];
                if (v < 0 || v >= hparams_.codebook_size) {
                    std::fprintf(stderr, "[eval_cached] codebook id out of range: cb=%d id=%d\n", cb, v);
                    return false;
                }
            }
        }
    }

    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;
    if (dim <= 0 || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 ||
        weights_.layers.empty() || !weights_.layers[0].wo) {
        std::fprintf(stderr, "[eval_cached] invalid attention dimensions/weights\n");
        return false;
    }

    // head_dim: from q_norm when qk_norm, else wo/head_count
    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        if (weights_.layers[0].q_norm->ne[0] <= 0 ||
            weights_.layers[0].q_norm->ne[0] > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        if (weights_.layers[0].wo->ne[0] <= 0 ||
            weights_.layers[0].wo->ne[0] > std::numeric_limits<int32_t>::max() ||
            weights_.layers[0].wo->ne[0] % n_head != 0) {
            return false;
        }
        head_dim = static_cast<int32_t>(weights_.layers[0].wo->ne[0] / n_head);
    }

    const int64_t q_size64  = static_cast<int64_t>(n_head) * head_dim;
    const int64_t kv_size64 = static_cast<int64_t>(n_head_kv) * head_dim;
    if (head_dim <= 0 || q_size64 <= 0 || kv_size64 <= 0 ||
        q_size64 > std::numeric_limits<int32_t>::max() ||
        kv_size64 > std::numeric_limits<int32_t>::max() ||
        q_size64 > std::numeric_limits<int64_t>::max() - 2 * kv_size64) {
        std::fprintf(stderr, "[eval_cached] attention shape overflow\n");
        return false;
    }
    const int32_t q_size   = static_cast<int32_t>(q_size64);
    const int32_t kv_size  = static_cast<int32_t>(kv_size64);
    const int64_t qkv_size = q_size64 + 2 * kv_size64;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float sem_scale  = 1.0f / std::sqrt(static_cast<float>(codebook_dim));

    // Prepare host-side input arrays
    std::vector<int32_t> semantic_vals(n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    std::vector<float>   semantic_mask_vals(n_tokens, 0.0f);
    std::vector<float>   token_scale_vals;
    std::vector<std::vector<int32_t>> cb_vals(hparams_.num_codebooks, std::vector<int32_t>(n_tokens, 0));
    std::vector<float> host_embedding_vals;
    std::vector<float> host_codebook_sum_vals;

    const bool use_host_semantic = cuda_host_embeddings_;
    const bool use_host_codebooks = cuda_host_codebook_embeddings_;
    if (use_host_semantic &&
        (!host_embeddings_.valid() || host_embeddings_.width != dim ||
         host_embeddings_.rows < hparams_.vocab_size)) {
        std::fprintf(stderr, "[eval_cached] CUDA host semantic embedding table is incomplete/incompatible\n");
        return false;
    }
    if (use_host_codebooks &&
        (!host_codebook_embeddings_.valid() || host_codebook_embeddings_.width != dim ||
         host_codebook_embeddings_.rows <
             static_cast<int64_t>(hparams_.num_codebooks) * hparams_.codebook_size)) {
        std::fprintf(stderr, "[eval_cached] CUDA host codebook embedding table is incomplete/incompatible\n");
        return false;
    }

    if (hparams_.scale_codebook_embeddings) {
        token_scale_vals.resize(n_tokens);
    }

    bool has_semantic = false;
    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t semantic = flat_tokens[t * codebook_dim];
        const bool is_semantic = (semantic >= hparams_.semantic_begin_id &&
                                  semantic <= hparams_.semantic_end_id);

        semantic_vals[t]      = semantic;
        pos_vals[t]           = n_past_ + t;
        semantic_mask_vals[t] = is_semantic ? 1.0f : 0.0f;
        has_semantic = has_semantic || is_semantic;

        if (!token_scale_vals.empty()) {
            token_scale_vals[t] = is_semantic ? sem_scale : 1.0f;
        }

        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            if (!is_semantic) continue;
            const int32_t v = flat_tokens[t * codebook_dim + cb + 1];
            cb_vals[cb][t] = v + cb * hparams_.codebook_size;
        }
    }

    if (use_host_semantic || (use_host_codebooks && has_semantic)) {
        const uint64_t host_elems = static_cast<uint64_t>(dim) * static_cast<uint64_t>(n_tokens);
        if (host_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
            return false;
        }
        if (use_host_semantic) {
            host_embedding_vals.resize(static_cast<size_t>(host_elems));
            for (int32_t t = 0; t < n_tokens; ++t) {
                float * dst = host_embedding_vals.data() + static_cast<size_t>(t) * dim;
                if (!decode_host_embedding_row(host_embeddings_, semantic_vals[t], dst)) {
                    std::fprintf(stderr, "[eval_cached] failed to decode semantic embedding row %d\n", semantic_vals[t]);
                    return false;
                }
            }
        }
        if (use_host_codebooks && has_semantic) {
            // If semantic embeddings are also host-side, accumulate codebooks
            // directly into that same activation buffer. Otherwise build one
            // separate F32 codebook-sum input; no full table is dequantized.
            if (!use_host_semantic) {
                host_codebook_sum_vals.assign(static_cast<size_t>(host_elems), 0.0f);
            }
            std::vector<float> row_tmp(static_cast<size_t>(dim));
            for (int32_t t = 0; t < n_tokens; ++t) {
                if (semantic_mask_vals[t] == 0.0f) continue;
                float * dst = use_host_semantic
                    ? host_embedding_vals.data() + static_cast<size_t>(t) * dim
                    : host_codebook_sum_vals.data() + static_cast<size_t>(t) * dim;
                for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
                    const int64_t row = static_cast<int64_t>(cb_vals[cb][t]);
                    if (!decode_host_embedding_row(host_codebook_embeddings_, row, row_tmp.data())) {
                        std::fprintf(stderr, "[eval_cached] failed to decode codebook embedding row %lld\n",
                                     static_cast<long long>(row));
                        return false;
                    }
                    for (int32_t i = 0; i < dim; ++i) dst[i] += row_tmp[static_cast<size_t>(i)];
                }
            }
        }
    }

    // Build computation graph. Reuse one arena per model instance. The KV-backed
    // model is not re-entrant, and the server serializes inference, so a
    // thread_local arena only multiplies retained host memory across workers.
    constexpr size_t ctx_size = 10u * 1024u * 1024u;
    if (graph_ctx_buf_.size() != ctx_size) graph_ctx_buf_.resize(ctx_size);
    ggml_init_params p = { ctx_size, graph_ctx_buf_.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;
    GgmlContextGuard ctx_guard(ctx0);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor * semantic_ids   = nullptr;
    ggml_tensor * positions      = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * semantic_mask  = nullptr;
    ggml_tensor * token_scale    = nullptr;
    std::vector<ggml_tensor *> cb_id_tensors(hparams_.num_codebooks, nullptr);
    ggml_tensor * host_embedding_input = nullptr;
    ggml_tensor * host_codebook_input = nullptr;
    ggml_tensor * causal_mask = nullptr;
    std::vector<float> causal_mask_vals;

    if (active_backend_is_metal(backend_)) {
        const int64_t total_k = static_cast<int64_t>(n_past_) + n_tokens;
        const uint64_t mask_elems = static_cast<uint64_t>(total_k) * static_cast<uint64_t>(n_tokens);
        if (total_k <= 0 || mask_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
            return false;
        }
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, total_k, n_tokens);
        causal_mask_vals.resize(static_cast<size_t>(mask_elems));
        for (int32_t q = 0; q < n_tokens; ++q) {
            const int64_t max_k = static_cast<int64_t>(n_past_) + q;
            for (int64_t k = 0; k < total_k; ++k) {
                causal_mask_vals[static_cast<size_t>(q) * static_cast<size_t>(total_k) + static_cast<size_t>(k)] =
                    (k <= max_k) ? 0.0f : -1.0e9f;
            }
        }
    }

    ggml_tensor * x = nullptr;
    if (use_host_semantic) {
        host_embedding_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens);
        x = host_embedding_input;
    } else {
        semantic_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        x = ggml_get_rows(ctx0, weights_.embeddings, semantic_ids);
        if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx0, x, GGML_TYPE_F32);
    }

    ggml_tensor * codebook_sum = nullptr;
    if (use_host_codebooks) {
        if (!use_host_semantic && has_semantic) {
            host_codebook_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens);
            codebook_sum = host_codebook_input;
        }
        // When both tables are host-side, codebook rows were already accumulated
        // into host_embedding_vals above, so no extra graph add is needed.
    } else {
        semantic_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            cb_id_tensors[cb] = ids;
            ggml_tensor * emb = ggml_get_rows(ctx0, weights_.codebook_embeddings, ids);
            if (emb->type != GGML_TYPE_F32) emb = ggml_cast(ctx0, emb, GGML_TYPE_F32);
            codebook_sum = (codebook_sum == nullptr)
                ? emb
                : add_same_shape_checked(ctx0, codebook_sum, emb, "add:codebook_sum");
        }
        if (codebook_sum != nullptr) {
            codebook_sum = ggml_mul(ctx0, codebook_sum,
                                    repeat_checked(ctx0, semantic_mask, codebook_sum, "repeat:semantic_mask"));
        }
    }

    if (codebook_sum != nullptr) {
        x = add_same_shape_checked(ctx0, x, codebook_sum, "add:semantic_codebooks");
    }
    if (hparams_.scale_codebook_embeddings) {
        token_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        x = ggml_mul(ctx0, x, repeat_checked(ctx0, token_scale, x, "repeat:token_scale"));
    }

    for (int32_t il = 0; il < hparams_.block_count; ++il) {
        const auto & layer = weights_.layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:wqkv");
        if (qkv->ne[0] != qkv_size || qkv->ne[1] != n_tokens || qkv->ne[2] != 1 || qkv->ne[3] != 1) {
            throw std::runtime_error("wqkv: unexpected output shape");
        }
        const size_t elem_size = ggml_element_size(qkv);
        const uint64_t k_offset_elems = static_cast<uint64_t>(q_size64);
        const uint64_t v_offset_elems = static_cast<uint64_t>(q_size64) + static_cast<uint64_t>(kv_size64);
        if (elem_size == 0 ||
            k_offset_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / elem_size) ||
            v_offset_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / elem_size)) {
            throw std::runtime_error("wqkv: view offset overflow");
        }
        const size_t k_offset = static_cast<size_t>(k_offset_elems) * elem_size;
        const size_t v_offset = static_cast<size_t>(v_offset_elems) * elem_size;

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], k_offset);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], v_offset);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        // QK norm (applied before RoPE)
        if (hparams_.attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.rms_norm_eps);
        }

        // RoPE
        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        // Write K/V into KV cache
        const size_t layer_off_k = static_cast<size_t>(il) * memory_k_->nb[3];
        const size_t layer_off_v = static_cast<size_t>(il) * memory_v_->nb[3];
        const size_t token_off_k = static_cast<size_t>(n_past_) * memory_k_->nb[2];
        const size_t token_off_v = static_cast<size_t>(n_past_) * memory_v_->nb[2];

        ggml_tensor * k_slot = ggml_view_3d(ctx0, memory_k_,
            head_dim, n_head_kv, n_tokens,
            memory_k_->nb[1], memory_k_->nb[2],
            layer_off_k + token_off_k);
        ggml_tensor * v_slot = ggml_view_3d(ctx0, memory_v_,
            head_dim, n_head_kv, n_tokens,
            memory_v_->nb[1], memory_v_->nb[2],
            layer_off_v + token_off_v);
        ggml_build_forward_expand(gf, cpy_same_shape_checked(ctx0, k, k_slot, "cpy:k_cache"));
        ggml_build_forward_expand(gf, cpy_same_shape_checked(ctx0, v, v_slot, "cpy:v_cache"));

        ggml_tensor * k_mem = k;
        ggml_tensor * v_mem = v;
        if (n_past_ > 0) {
            ggml_tensor * k_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_k_, static_cast<int64_t>(n_past_) * kv_size, layer_off_k),
                head_dim, n_head_kv, n_past_);
            ggml_tensor * v_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_v_, static_cast<int64_t>(n_past_) * kv_size, layer_off_v),
                head_dim, n_head_kv, n_past_);
            if (k_past->type != k->type) k_past = ggml_cast(ctx0, k_past, k->type);
            if (v_past->type != v->type) v_past = ggml_cast(ctx0, v_past, v->type);
            k_mem = ggml_concat(ctx0, k_past, k, 2);
            v_mem = ggml_concat(ctx0, v_past, v, 2);
        }

        if (n_head != n_head_kv && q->type != GGML_TYPE_F32) {
            q = ggml_cast(ctx0, q, GGML_TYPE_F32);
        }
        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k_mem, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v_mem, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:kq");
        ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
        ggml_tensor * KQm = nullptr;
        if (causal_mask) {
            KQm = add_same_shape_checked(
                ctx0, KQs, repeat_checked(ctx0, causal_mask, KQs, "repeat:causal_mask"),
                "add:causal_mask");
        } else {
            KQm = ggml_diag_mask_inf(ctx0, KQs, n_past_);
        }
        ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = cpy_same_shape_checked(
            ctx0, KQVm, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens), "cpy:attn");
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:wo");

        ggml_tensor * h     = add_same_shape_checked(ctx0, x, attn_out, "add:attention_residual");
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:w3");
        ggml_tensor * ff_h  = swiglu_split_checked(ctx0, gate, up, "swiglu:ffn");
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:w2");

        x = add_same_shape_checked(ctx0, h, ff_out, "add:ffn_residual");
    }

    ggml_tensor * slow_out  = rms_norm_weighted(ctx0, x, weights_.norm, hparams_.rms_norm_eps);
    ggml_tensor * slow_cont = ggml_cont(ctx0, slow_out);
    ggml_tensor * hidden_last = cpy_same_shape_checked(
        ctx0, last_token_view(ctx0, slow_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1), "cpy:hidden_last");

    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.output, hidden_last, "mul_mat:logits");
    ggml_build_forward_expand(gf, logits);

    // Allocate and run
    if (!ggml_gallocr_alloc_graph(allocr_, gf)) {
        std::fprintf(stderr, "[eval_cached] gallocr alloc failed\n");
        return false;
    }

    ggml_backend_tensor_set(positions, pos_vals.data(), 0, n_tokens * sizeof(int32_t));
    if (host_embedding_input) {
        ggml_backend_tensor_set(host_embedding_input, host_embedding_vals.data(), 0,
                                host_embedding_vals.size() * sizeof(float));
    } else if (semantic_ids) {
        ggml_backend_tensor_set(semantic_ids, semantic_vals.data(), 0, n_tokens * sizeof(int32_t));
    }
    if (host_codebook_input) {
        ggml_backend_tensor_set(host_codebook_input, host_codebook_sum_vals.data(), 0,
                                host_codebook_sum_vals.size() * sizeof(float));
    }
    if (semantic_mask) {
        ggml_backend_tensor_set(semantic_mask, semantic_mask_vals.data(), 0, n_tokens * sizeof(float));
    }
    if (token_scale) {
        ggml_backend_tensor_set(token_scale, token_scale_vals.data(), 0, n_tokens * sizeof(float));
    }
    if (causal_mask) {
        ggml_backend_tensor_set(causal_mask, causal_mask_vals.data(), 0,
                                causal_mask_vals.size() * sizeof(float));
    }
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        if (cb_id_tensors[cb]) {
            ggml_backend_tensor_set(cb_id_tensors[cb], cb_vals[cb].data(), 0, n_tokens * sizeof(int32_t));
        }
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[eval_cached] compute failed\n");
        return false;
    }

    result.hidden.resize(dim);
    result.logits.resize(hparams_.vocab_size);
    ggml_backend_tensor_get(hidden_last, result.hidden.data(), 0, dim * sizeof(float));
    ggml_backend_tensor_get(logits,      result.logits.data(), 0, hparams_.vocab_size * sizeof(float));

    n_past_ += n_tokens;
    return true;
}

// ---------------------------------------------------------------------------
// fast_decode() -- fast AR decoder (matches eval_fast_prefix from reference)
// ---------------------------------------------------------------------------

bool SlowARModel::fast_decode(const std::vector<float> & hidden_in,
                               const std::vector<int32_t> & prefix_tokens,
                               int32_t n_threads,
                               std::vector<float> & logits_out) {
    if (!hparams_.has_fast_decoder) {
        std::fprintf(stderr, "[fast_decode] model has no fast decoder\n");
        return false;
    }
    if (hparams_.embedding_length < 0 ||
        hidden_in.size() != static_cast<size_t>(hparams_.embedding_length)) {
        std::fprintf(stderr, "[fast_decode] expected hidden size %d, got %zu\n",
            hparams_.embedding_length, hidden_in.size());
        return false;
    }
    if (hparams_.num_codebooks <= 0 || hparams_.fast_context_length <= 0 ||
        prefix_tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        prefix_tokens.size() >= static_cast<size_t>(hparams_.num_codebooks) ||
        prefix_tokens.size() + 1u > static_cast<size_t>(hparams_.fast_context_length)) {
        std::fprintf(stderr, "[fast_decode] prefix too long (%zu)\n", prefix_tokens.size());
        return false;
    }
    for (int32_t token : prefix_tokens) {
        if (token < 0 || token >= hparams_.codebook_size) {
            std::fprintf(stderr, "[fast_decode] codebook id out of range: %d\n", token);
            return false;
        }
    }

    const int32_t fast_dim  = hparams_.fast_embedding_length;
    const int32_t n_head    = hparams_.fast_head_count;
    const int32_t n_head_kv = hparams_.fast_head_count_kv;
    if (fast_dim <= 0 || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0) {
        std::fprintf(stderr, "[fast_decode] invalid attention dimensions\n");
        return false;
    }
    const int32_t head_dim  = (hparams_.fast_head_dim > 0)
                                ? hparams_.fast_head_dim
                                : fast_dim / n_head;
    const int64_t q_size64  = static_cast<int64_t>(n_head) * head_dim;
    const int64_t kv_size64 = static_cast<int64_t>(n_head_kv) * head_dim;
    if (head_dim <= 0 || q_size64 <= 0 || kv_size64 <= 0 ||
        q_size64 > std::numeric_limits<int32_t>::max() ||
        kv_size64 > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "[fast_decode] attention shape overflow\n");
        return false;
    }
    const int32_t q_size    = static_cast<int32_t>(q_size64);
    const int32_t kv_size   = static_cast<int32_t>(kv_size64);
    const int64_t qkv_size  = q_size64 + 2 * kv_size64;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int32_t n_tokens  = static_cast<int32_t>(prefix_tokens.size()) + 1;

    std::vector<float> host_prefix_embeddings;
    const bool use_host_fast_embeddings = cuda_fast_host_embeddings_ && !prefix_tokens.empty();
    if (use_host_fast_embeddings) {
        if (!host_fast_embeddings_.valid() || host_fast_embeddings_.width != fast_dim ||
            host_fast_embeddings_.rows < hparams_.codebook_size) {
            std::fprintf(stderr, "[fast_decode] CUDA host fast-embedding table is incomplete/incompatible\n");
            return false;
        }
        const uint64_t elems = static_cast<uint64_t>(fast_dim) * prefix_tokens.size();
        if (elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) return false;
        host_prefix_embeddings.resize(static_cast<size_t>(elems));
        for (size_t i = 0; i < prefix_tokens.size(); ++i) {
            float * dst = host_prefix_embeddings.data() + i * static_cast<size_t>(fast_dim);
            if (!decode_host_embedding_row(host_fast_embeddings_, prefix_tokens[i], dst)) {
                std::fprintf(stderr, "[fast_decode] failed to decode fast embedding row %d\n", prefix_tokens[i]);
                return false;
            }
        }
    }

    constexpr size_t fast_ctx_size = 8u * 1024u * 1024u;
    if (fast_graph_ctx_buf_.size() != fast_ctx_size) fast_graph_ctx_buf_.resize(fast_ctx_size);
    ggml_init_params p = { fast_ctx_size, fast_graph_ctx_buf_.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;
    GgmlContextGuard ctx_guard(ctx0);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // hidden input (from slow decoder)
    ggml_tensor * hidden0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.embedding_length, 1);

    // Optional project_in (not present for this model, but handle it)
    ggml_tensor * projected = (weights_.fast_project_in != nullptr)
        ? mul_mat_checked(ctx0, weights_.fast_project_in, hidden0, "mul_mat:fast_project_in")
        : hidden0;
    if (projected->type != GGML_TYPE_F32) {
        projected = ggml_cast(ctx0, projected, GGML_TYPE_F32);
    }
    if (projected->ne[0] != fast_dim || projected->ne[1] != 1 ||
        projected->ne[2] != 1 || projected->ne[3] != 1) {
        throw std::runtime_error("fast_project_in: unexpected output shape");
    }

    // Build sequence: [projected_hidden; prefix_embeddings]
    ggml_tensor * x = projected;
    ggml_tensor * prefix_ids = nullptr;
    ggml_tensor * prefix_embedding_input = nullptr;
    if (!prefix_tokens.empty()) {
        ggml_tensor * prefix_emb = nullptr;
        if (use_host_fast_embeddings) {
            prefix_embedding_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fast_dim,
                                                        static_cast<int64_t>(prefix_tokens.size()));
            prefix_emb = prefix_embedding_input;
        } else {
            prefix_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)prefix_tokens.size());
            prefix_emb = ggml_get_rows(ctx0, weights_.fast_embeddings, prefix_ids);
            if (prefix_emb->type != GGML_TYPE_F32) {
                prefix_emb = ggml_cast(ctx0, prefix_emb, GGML_TYPE_F32);
            }
        }
        if (prefix_emb->ne[0] != fast_dim ||
            prefix_emb->ne[1] != static_cast<int64_t>(prefix_tokens.size()) ||
            prefix_emb->ne[2] != 1 || prefix_emb->ne[3] != 1) {
            throw std::runtime_error("fast_embeddings: unexpected output shape");
        }
        x = ggml_concat(ctx0, x, prefix_emb, 1);
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos_vals[i] = i;
    ggml_tensor * causal_mask = nullptr;
    std::vector<float> causal_mask_vals;
    if (active_backend_is_metal(backend_)) {
        const uint64_t mask_elems = static_cast<uint64_t>(n_tokens) * static_cast<uint64_t>(n_tokens);
        if (mask_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
            return false;
        }
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
        causal_mask_vals.resize(static_cast<size_t>(mask_elems));
        for (int32_t q = 0; q < n_tokens; ++q) {
            for (int32_t k = 0; k < n_tokens; ++k) {
                causal_mask_vals[static_cast<size_t>(q) * n_tokens + k] = (k <= q) ? 0.0f : -1.0e9f;
            }
        }
    }

    for (int32_t il = 0; il < hparams_.fast_block_count; ++il) {
        const auto & layer = weights_.fast_layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:fast_wqkv");
        if (qkv->ne[0] != qkv_size || qkv->ne[1] != n_tokens || qkv->ne[2] != 1 || qkv->ne[3] != 1) {
            throw std::runtime_error("fast_wqkv: unexpected output shape");
        }
        const size_t elem_size = ggml_element_size(qkv);
        const uint64_t k_offset_elems = static_cast<uint64_t>(q_size64);
        const uint64_t v_offset_elems = static_cast<uint64_t>(q_size64) + static_cast<uint64_t>(kv_size64);
        if (elem_size == 0 ||
            k_offset_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / elem_size) ||
            v_offset_elems > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / elem_size)) {
            throw std::runtime_error("fast_wqkv: view offset overflow");
        }
        const size_t k_offset = static_cast<size_t>(k_offset_elems) * elem_size;
        const size_t v_offset = static_cast<size_t>(v_offset_elems) * elem_size;

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], k_offset);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], v_offset);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.fast_attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.fast_rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.fast_rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:fast_kq");
        ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
        ggml_tensor * KQm = nullptr;
        if (causal_mask) {
            KQm = add_same_shape_checked(
                ctx0, KQs, repeat_checked(ctx0, causal_mask, KQs, "repeat:fast_causal_mask"),
                "add:fast_causal_mask");
        } else {
            KQm = ggml_diag_mask_inf(ctx0, KQs, 0);
        }
        ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:fast_kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = cpy_same_shape_checked(
            ctx0, KQVm, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens), "cpy:fast_attn");
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:fast_wo");

        ggml_tensor * h     = add_same_shape_checked(ctx0, x, attn_out, "add:fast_attention_residual");
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:fast_w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:fast_w3");
        ggml_tensor * ff_h  = swiglu_split_checked(ctx0, gate, up, "swiglu:fast_ffn");
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:fast_w2");

        x = add_same_shape_checked(ctx0, h, ff_out, "add:fast_ffn_residual");
    }

    ggml_tensor * fast_out  = rms_norm_weighted(ctx0, x, weights_.fast_norm, hparams_.fast_rms_norm_eps);
    ggml_tensor * fast_cont = ggml_cont(ctx0, fast_out);
    ggml_tensor * fast_last = cpy_same_shape_checked(
        ctx0, last_token_view(ctx0, fast_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fast_dim, 1), "cpy:fast_last");
    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.fast_output, fast_last, "mul_mat:fast_logits");
    ggml_build_forward_expand(gf, logits);

    if (!ggml_gallocr_alloc_graph(fast_allocr_, gf)) {
        std::fprintf(stderr, "[fast_decode] gallocr alloc failed\n");
        return false;
    }

    ggml_backend_tensor_set(hidden0,   hidden_in.data(),    0, hidden_in.size() * sizeof(float));
    ggml_backend_tensor_set(positions, pos_vals.data(),     0, pos_vals.size() * sizeof(int32_t));
    if (prefix_embedding_input) {
        ggml_backend_tensor_set(prefix_embedding_input, host_prefix_embeddings.data(), 0,
                                host_prefix_embeddings.size() * sizeof(float));
    } else if (prefix_ids) {
        ggml_backend_tensor_set(prefix_ids, prefix_tokens.data(), 0,
                                prefix_tokens.size() * sizeof(int32_t));
    }
    if (causal_mask) {
        ggml_backend_tensor_set(causal_mask, causal_mask_vals.data(), 0,
                                causal_mask_vals.size() * sizeof(float));
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fast_decode] compute failed\n");
        return false;
    }

    // logits_out has codebook_size elements (NOT vocab_size)
    logits_out.resize(hparams_.codebook_size);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, hparams_.codebook_size * sizeof(float));

    return true;
}

} // namespace s2

// ---------------------------------------------------------------------------
// free_kv_cache() -- implementacion anadida por s2.cpp fork
// Libera el buffer de KV cache de VRAM/RAM para dar espacio al codec decode.
// Declarada en s2_model.h pero no implementada en el original de mach92432.
// ---------------------------------------------------------------------------
namespace s2 {
void SlowARModel::free_kv_cache() {
    if (kv_buf_) {
        ggml_backend_buffer_free(kv_buf_);
        kv_buf_ = nullptr;
    }
    if (ctx_kv_) {
        ggml_free(ctx_kv_);
        ctx_kv_ = nullptr;
    }
    memory_k_ = nullptr;
    memory_v_ = nullptr;
    max_seq_len_ = 0;
    n_past_ = 0;
}
} // namespace s2
