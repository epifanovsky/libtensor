#ifndef LIBTENSOR_TO_MULT_H
#define LIBTENSOR_TO_MULT_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Element-wise multiplication and division
    \tparam N Tensor order.

    The operation multiplies or divides two tensors element by element.
    Both arguments and result must have the same dimensions or an exception
    will be thrown. When the division is requested, no checks are performed
    to ensure that the denominator is non-zero.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_mult : public timings< to_mult<N, T> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, T> &m_ta; //!< First argument
    dense_tensor_rd_i<N, T> &m_tb; //!< Second argument
    permutation<N> m_perma; //!< Tensor transformation of first argument
    permutation<N> m_permb; //!< Tensor transformation of second argument
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    T m_c; //!< Scaling coefficient
    dimensions<N> m_dimsc; //!< Result dimensions

public:
    /** \brief Initializes the operation
        \param ta First argument.
        \param tra Tensor transformation of ta with respect to result.
        \param tb Second argument.
        \param trb Tensor transformation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation
     **/
    to_mult(
            dense_tensor_rd_i<N, T> &ta,
            const tensor_transf<N, T> &tra,
            dense_tensor_rd_i<N, T> &tb,
            const tensor_transf<N, T> &trb,
            bool recip, const scalar_transf<T> &trc =
                    scalar_transf<T>());

    /** \brief Creates the operation
        \param ta First argument.
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param coeff Scaling coefficient
     **/
    to_mult(dense_tensor_rd_i<N, T> &ta, dense_tensor_rd_i<N, T> &tb,
        bool recip = false, T c = 1.0);

    /** \brief Initializes the operation
        \param ta First argument.
        \param pa Permutation of ta with respect to result.
        \param tb Second argument.
        \param pb Permutation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation
     **/
    to_mult(
            dense_tensor_rd_i<N, T> &ta,
            const permutation<N> &pa,
            dense_tensor_rd_i<N, T> &tb,
            const permutation<N> &pb,
            bool recip, T c = 1.0);

    void prefetch();

    void perform(bool zero, dense_tensor_wr_i<N, T> &tc);
};


} // namespace libtensor

#endif // LIBTENSOR_TO_MULT_H
