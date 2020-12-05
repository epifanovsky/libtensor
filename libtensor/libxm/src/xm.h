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

/** \file
 *  \brief Main libxm include file and tensor operations. */

#ifdef __cplusplus
extern "C" {
#endif

/** Print libxm banner to the standard output. */
void xm_print_banner(void);

/** Set all elements of a tensor to the same value.
 *  \param a Input tensor.
 *  \param x Scalar value. */
void xm_set(xm_tensor_t *a, xm_scalar_t x);

/** Copy tensor data while multiplying by a scaling factor (a = s * b).
 *  Tensors can have different scalar types.
 *  Tensors must have compatible block-structures.
 *  This function does not change the original block-structure of the output
 *  tensor.
 *  \param a Output tensor.
 *  \param s Scaling factor.
 *  \param b Input tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *
 *  \code
 *  Example: xm_copy(a, 2.0, b, "ijk", "kij");
 *           a_ijk = 2 * b_kij
 *  \endcode */
void xm_copy(xm_tensor_t *a, xm_scalar_t s, const xm_tensor_t *b,
    const char *idxa, const char *idxb);

/** Add tensors (a = alpha * a + beta * b). Tensors must have compatible
 *  block-structures. This function does not change the block-structure of the
 *  output tensor.
 *  \param alpha Scalar factor.
 *  \param a First tensor.
 *  \param beta Scalar factor.
 *  \param b Second tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *
 *  \code
 *  Example: xm_add(1.0, a, 2.0, b, "ij", "ji");
 *           a_ij = a_ij + 2 * b_ji
 *  \endcode */
void xm_add(xm_scalar_t alpha, xm_tensor_t *a, xm_scalar_t beta,
    const xm_tensor_t *b, const char *idxa, const char *idxb);

/** Multiply tensors element-wise. Tensors must have compatible
 *  block-structures.
 *  This function does not change the block-structure of the output tensor.
 *  \param a First tensor.
 *  \param b Second tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *
 *  \code
 *  Example: xm_mul(a, b, "ij", "ji");
 *           a_ij = a_ij * b_ji
 *  \endcode */
void xm_mul(xm_tensor_t *a, const xm_tensor_t *b, const char *idxa,
    const char *idxb);

/** Divide tensors element-wise. Tensors must have compatible block-structures.
 *  This function does not change the block-structure of the output tensor.
 *  \param a First tensor.
 *  \param b Second tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *
 *  \code
 *  Example: xm_div(a, b, "ij", "ji");
 *           a_ij = a_ij / b_ji
 *  \endcode */
void xm_div(xm_tensor_t *a, const xm_tensor_t *b, const char *idxa,
    const char *idxb);

/** Dot product of two tensors. Tensors must have compatible block-structures.
 *  \param a First tensor.
 *  \param b Second tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *  \return Dot product of two tensors.
 *
 *  \code
 *  Example: dot = xm_dot(a, b, "ij", "ij");
 *           dot = a * b
 *  \endcode */
xm_scalar_t xm_dot(const xm_tensor_t *a, const xm_tensor_t *b,
    const char *idxa, const char *idxb);

/** Contract tensors over contraction indices specified by strings \p idxa and
 *  \p idxb (c = alpha * a * b + beta * c). Permutation of tensor \p c is
 *  specified by \p idxc. The routine will perform optimal contraction using
 *  symmetry and sparsity information obtained from tensors' block-structures.
 *  Tensors must be setup beforehand so that they have correct symmetries.
 *  This function does not change the original block-structure of the output
 *  tensor.
 *  \param alpha Scalar factor.
 *  \param a First tensor.
 *  \param b Second tensor.
 *  \param beta Scalar factor.
 *  \param c Third tensor.
 *  \param idxa Indices of \p a.
 *  \param idxb Indices of \p b.
 *  \param idxc Indices of \p c.
 *
 *  \code
 *  Example: xm_contract(1.0, a, b, 2.0, c, "abcd", "ijcd", "ijab");
 *           c_ijab = a_abcd * b_ijcd + 2 * c_ijab
 *  \endcode */
void xm_contract(xm_scalar_t alpha, const xm_tensor_t *a, const xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
