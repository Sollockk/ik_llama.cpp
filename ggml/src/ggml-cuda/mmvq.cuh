//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "mmvq-args.h"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Direct MMVQ dispatch by type from pre-built args (for GPU delta correction).
void ggml_cuda_mmvq_dispatch(ggml_type type, const mmvq_args & args, cudaStream_t stream);

void ggml_cuda_op_mul_mat_vec_q_biased(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_mmvq_type_supported(ggml_type src0_type);

void ggml_cuda_op_mul_mat_vec_q_3D(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_mul_mat_vec_q_id(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Fused base+delta GEMV: single kernel for IQ2_S/IQ3_S base + QD4_K delta.
struct mmvq_fused_delta_args;
void mul_mat_vec_iq2s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
void mul_mat_vec_iq3s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);

void ggml_cuda_op_fused_mul_mat_vec_q_id(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias_u, const ggml_tensor * bias_g,
    const char * src0_dd_u, const char * src0_dd_g, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, ggml_unary_op unary_op, float limit, cudaStream_t stream);
