#ifndef LIBTENSOR_TOD_MULT_H
#define LIBTENSOR_TOD_MULT_H

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
template<size_t N>
class tod_mult : public timings< tod_mult<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< First argument
    dense_tensor_rd_i<N, double> &m_tb; //!< Second argument
    tensor_transf<N, double> m_tra; //!< Tensor transformation of first argument
    tensor_transf<N, double> m_trb; //!< Tensor transformation of second argument
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    scalar_transf<double> m_trc; //!< Scaling coefficient
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
    tod_mult(
            dense_tensor_rd_i<N, double> &ta,
            const tensor_transf<N, double> &tra,
            dense_tensor_rd_i<N, double> &tb,
            const tensor_transf<N, double> &trb,
            bool recip, const scalar_transf<double> &trc =
                    scalar_transf<double>());

    /** \brief Creates the operation
        \param ta First argument.
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param coeff Scaling coefficient
     **/
    tod_mult(dense_tensor_rd_i<N, double> &ta, dense_tensor_rd_i<N, double> &tb,
        bool recip = false, double c = 1.0);

    /** \brief Initializes the operation
        \param ta First argument.
        \param pa Permutation of ta with respect to result.
        \param tb Second argument.
        \param pb Permutation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation
     **/
    tod_mult(
            dense_tensor_rd_i<N, double> &ta,
            const permutation<N> &pa,
            dense_tensor_rd_i<N, double> &tb,
            const permutation<N> &pb,
            bool recip, double c = 1.0);

    void prefetch();

    void perform(bool zero, dense_tensor_wr_i<N, double> &tc);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT_H
