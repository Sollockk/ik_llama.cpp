#pragma once

#include <cstdint>

struct mmvq_sparse_args {
    const void *    packed_weights;   // VRAM: packed non-zero blocks
    const uint16_t * block_index;    // VRAM: uint16 indices [n_blocks_stored]
    const void *    vy;              // Q8_1 quantized input
    float *         dst;             // output (accumulate: dst += sparse_delta @ input)
    int             ncols_x;         // full row width in elements (ne00)
    int             nrows_x;         // number of output rows
    int             nrows_y;         // padded input rows (for Q8_1 block alignment)
    int             nrows_dst;       // destination rows
    int             n_blocks_stored; // number of non-zero blocks per row
    int             packed_row_blocks;// same as n_blocks_stored
};
