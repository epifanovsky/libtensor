#ifndef LIBTENSOR_DIAG_TOD_DOTPROD_H
#define LIBTENSOR_DIAG_TOD_DOTPROD_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Computes the dot product of two diagonal tensors
    \param N Tensor order.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_dotprod :
    public timings< diag_tod_dotprod<N> >, public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    diag_tensor_rd_i<N, double> &m_dta; //!< First tensor (A)
    diag_tensor_rd_i<N, double> &m_dtb; //!< Second tensor (B)
    tensor_transf<N, double> m_tra; //!< Transformation of A
    tensor_transf<N, double> m_trb; //!< Transformation of B

public:
    /** \brief Initializes the operation
        \param dta First tensor (A)
        \param dtb Second tensor (B)
     **/
    diag_tod_dotprod(
        diag_tensor_rd_i<N, double> &dta,
        diag_tensor_rd_i<N, double> &dtb);

    /** \brief Initializes the operation
        \param dta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param dtb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    diag_tod_dotprod(
        diag_tensor_rd_i<N, double> &dta, const permutation<N> &perma,
        diag_tensor_rd_i<N, double> &dtb, const permutation<N> &permb);

    /** \brief Initializes the operation
        \param dta First tensor (A)
        \param tra Transformation of first tensor (A)
        \param dtb Second tensor (B)
        \param trb Transformation of second tensor (B)
     **/
    diag_tod_dotprod(
        diag_tensor_rd_i<N, double> &dta, const tensor_transf<N, double> &tra,
        diag_tensor_rd_i<N, double> &dtb, const tensor_transf<N, double> &trb);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Computes the dot product and returns the value
     **/
    double calculate();

private:
    /** \brief Returns true if the dimensions of A and B are compatible
     **/
    bool verify_dims();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_DOTPROD_H

