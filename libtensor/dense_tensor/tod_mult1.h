#ifndef LIBTENSOR_TOD_MULT1_H
#define LIBTENSOR_TOD_MULT1_H

#include <libtensor/timings.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Element-wise multiplication and division
    \tparam N Tensor order.

    The operation multiplies or divides two tensors element by element.

    \f[ a_i = a_i b_i \qquad a_i = \frac{a_i}{b_i} \f]
    \f[ a_i = a_i + c a_i b_i \qquad a_i = a_i + c \frac{a_i}{b_i} \f]

    The numerator and the result are the same tensor. Both tensors must
    have the same dimensions or an exception will be thrown. When
    the division is requested, no checks are performed to ensure that
    the denominator is non-zero.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_mult1 : public timings< tod_mult1<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_tb; //!< Second argument
    permutation<N> m_pb; //!< Permutation of argument
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    double m_c; //!< Scaling coefficient

public:
    /** \brief Creates the operation
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    tod_mult1(dense_tensor_rd_i<N, double> &tb, bool recip = false,
        double c = 1.0) :
        m_tb(tb), m_recip(recip), m_c(c)
    { }

    /** \brief Creates the operation
        \param tb Second argument.
        \param p Permutation of argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    tod_mult1(dense_tensor_rd_i<N, double> &tb, const permutation<N> &p,
        bool recip = false, double c = 1.0) :
        m_tb(tb), m_pb(p), m_recip(recip), m_c(c)
    { }

    /** \brief Performs the operation, replaces the output.
        \param ta Tensor A.
     **/
    void perform(dense_tensor_wr_i<N, double> &ta);

    /** \brief Performs the operation, adds to the output.
        \param ta Tensor A.
        \param c Coefficient.
     **/
    void perform(dense_tensor_wr_i<N, double> &ta, double c);

private:
    void do_perform(dense_tensor_wr_i<N, double> &ta, bool doadd, double c);

private:
    tod_mult1(const tod_mult1&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT1_H
