#include "../mmvq-fused-delta.cuh"

// Non-WHT QD1_K (sign-only, ~55-65% energy retention)
void mul_mat_vec_iq2s_qd1k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ2_S, GGML_TYPE_QD1_K, false>(args, stream);
}

void mul_mat_vec_iq3s_qd1k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ3_S, GGML_TYPE_QD1_K, false>(args, stream);
}

// WHT QD1_K (sign of WHT'd delta, ~85-95% energy retention at dp4a speed)
void mul_mat_vec_iq2s_qd1k_wht_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ2_S, GGML_TYPE_QD1_K, true>(args, stream);
}

void mul_mat_vec_iq3s_qd1k_wht_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ3_S, GGML_TYPE_QD1_K, true>(args, stream);
}
