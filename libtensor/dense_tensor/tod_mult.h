#ifndef LIBTENSOR_TOD_MULT_H
#define LIBTENSOR_TOD_MULT_H

#include <libtensor/timings.h>
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
class tod_mult : public timings< tod_mult<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< First argument
    dense_tensor_rd_i<N, double> &m_tb; //!< Second argument
    permutation<N> m_perma; //!< Permutation of first argument
    permutation<N> m_permb; //!< Permutation of second argument
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    double m_c; //!< Scaling coefficient
    dimensions<N> m_dimsc; //!< Result dimensions

public:
    /** \brief Creates the operation
        \param ta First argument.
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param coeff Scaling coefficient
     **/
    tod_mult(dense_tensor_rd_i<N, double> &ta, dense_tensor_rd_i<N, double> &tb,
        bool recip = false, double c = 1.0);

    /** \brief Creates the operation
        \param ta First argument.
        \param pa Permutation of ta with respect to result.
        \param tb Second argument.
        \param pb Permutation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param coeff Scaling coefficient
     **/
    tod_mult(dense_tensor_rd_i<N, double> &ta, const permutation<N> &pa,
            dense_tensor_rd_i<N, double> &tb, const permutation<N> &pb,
            bool recip = false, double c = 1.0);

    void prefetch();

    void perform(bool zero, double c, dense_tensor_wr_i<N, double> &tc);

    void perform(dense_tensor_wr_i<N, double> &tc);

    void perform(dense_tensor_wr_i<N, double> &tc, double c);

private:
    void do_perform(dense_tensor_wr_i<N, double> &tc, bool doadd, double c);

private:
    tod_mult(const tod_mult&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT_H
