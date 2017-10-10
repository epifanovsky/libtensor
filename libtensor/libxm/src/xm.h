/*
 * Copyright (c) 2014-2017 Ilya Kaliman
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef XM_H_INCLUDED
#define XM_H_INCLUDED

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Print libxm banner to the standard output. */
void xm_print_banner(void);

/* Set all canonical block elements of tensor "a" to value "x". */
void xm_set(xm_tensor_t *a, xm_scalar_t x);

/* Copy tensor block data from "b" to "a" while multiplying by a scaling factor
 * (a = s * b). Tensors must have compatible block-structures. Permutation is
 * specified using strings idxa and idxb. This function does not change the
 * original block-structure of the output tensor.
 *
 * Example: xm_copy(a, 2.0, b, "ijk", "kij");
 *          a_ijk = 2 * b_kij
 */
void xm_copy(xm_tensor_t *a, xm_scalar_t s, const xm_tensor_t *b,
    const char *idxa, const char *idxb);

/* Add tensors (a = alpha * a + beta * b). Tensors must have compatible
 * block-structures. Permutation is specified using strings idxa and idxb.
 * This function does not change the original block-structure of the output
 * tensor.
 *
 * Example: xm_add(1.0, a, 2.0, b, "ij", "ji");
 *          a_ij = a_ij + 2 * b_ji
 */
void xm_add(xm_scalar_t alpha, xm_tensor_t *a, xm_scalar_t beta,
    const xm_tensor_t *b, const char *idxa, const char *idxb);

/* Contract two tensors over contraction indices specified by strings idxa and
 * idxb (c = alpha * a * b + beta * c). Permutation of tensor "c" is specified
 * by idxc. The routine will perform optimal contraction using symmetry and
 * sparsity information obtained from tensors' block-structures. It is the
 * user's responsibility to setup all tensors so that they have correct
 * symmetries. This function does not change the original block-structure of
 * the output tensor.
 *
 * Example: xm_contract(1.0, a, b, 2.0, c, "abcd", "ijcd", "ijab");
 *          c_ijab = a_abcd * b_ijcd + 2 * c_ijab
 */
void xm_contract(xm_scalar_t alpha, const xm_tensor_t *a, const xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
