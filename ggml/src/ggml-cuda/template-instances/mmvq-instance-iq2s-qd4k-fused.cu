#include "../mmvq-fused-delta.cuh"

void mul_mat_vec_iq2s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ2_S, GGML_TYPE_QD4_K, false>(args, stream);
}

void mul_mat_vec_iq3s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    mul_mat_vec_fused_delta_cuda<GGML_TYPE_IQ3_S, GGML_TYPE_QD4_K, false>(args, stream);
}
