#ifndef LIBTENSOR_DIAG_TOD_MULT1_H
#define LIBTENSOR_DIAG_TOD_MULT1_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief In-place element-wise multiplication and division
    \param N Tensor order.

    The operation multiplies or divides two tensors element by element.

    \f[ a_i = a_i b_i \qquad a_i = \frac{a_i}{b_i} \f]
    \f[ a_i = a_i + c a_i b_i \qquad a_i = a_i + c \frac{a_i}{b_i} \f]

    The division operation is constrained to the allowed space of tensor A,
    in which B must be non-zero. If B has zero elements in the space of A,
    division by zero exception may occur.

    \sa tod_mult1

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_mult1 : public timings< diag_tod_mult1<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_rd_i<N, double> &m_dtb; //!< Tensor B
    tensor_transf<N, double> m_trb; //!< Tensor transformation of B
    bool m_recip; //!< Reciprocal flag (division if true)

public:
    /** \brief Initializes the operation
        \param dtb Tensor B.
        \param recip Reciprocal flag.
     **/
    diag_tod_mult1(
        diag_tensor_rd_i<N, double> &dtb,
        bool recip = false) :
        m_dtb(dtb), m_recip(recip)
    { }

    /** \brief Initializes the operation
        \param dtb Tensor B.
        \param trb Transformation of B.
        \param recip Reciprocal flag.
     **/
    diag_tod_mult1(
        diag_tensor_rd_i<N, double> &dtb,
        const tensor_transf<N, double> &trb,
        bool recip = false) :
        m_dtb(dtb), m_trb(trb), m_recip(recip)
    { }

    /** \brief Performs the operation
        \param zero Zero output before copying.
        \param dta Output tensor.
     **/
    void perform(bool zero, diag_tensor_wr_i<N, double> &dta);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_MULT1_H

