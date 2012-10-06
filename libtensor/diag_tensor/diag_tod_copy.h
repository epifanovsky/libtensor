#ifndef LIBTENSOR_DIAG_TOD_COPY_H
#define LIBTENSOR_DIAG_TOD_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Copies diag tensors with and without addition
    \param N Tensor order.

    Copies the structure and data of one diagonal tensor (source, A)
    to another (destination, B). If requested, also performs scaling and
    permutation.

    Without addition: the destination B is reset first, on output B is
    the exact copy of A.
    With addition: the space of B is modified if necessary to fit the result
    of (A + B).

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_copy : public timings< diag_tod_copy<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_rd_i<N, double> &m_dta; //!< Source tensor A
    tensor_transf<N, double> m_tra; //!< Tensor transformation of A

public:
    /** \brief Initializes the operation
        \param dta Source tensor A.
     **/
    diag_tod_copy(
        diag_tensor_rd_i<N, double> &dta) :
        m_dta(dta)
    { }

    /** \brief Initializes the operation
        \param dta Source tensor A.
        \param perma Permutation of A.
        \param ka Scaling factor of A.
     **/
    diag_tod_copy(
        diag_tensor_rd_i<N, double> &dta,
        const permutation<N> &perma,
        double ka) :
        m_dta(dta), m_tra(perma, scalar_transf<double>(ka))
    { }

    /** \brief Initializes the operation
        \param dta Source tensor A.
        \param tra Transformation of A.
     **/
    diag_tod_copy(
        diag_tensor_rd_i<N, double> &dta,
        const tensor_transf<N, double> &tra) :
        m_dta(dta), m_tra(tra)
    { }

    /** \brief Performs the operation
        \param zero Zero output before copying.
        \param c Scaling factor.
        \param dtb Output tensor.
     **/
    void perform(bool zero, double c, diag_tensor_wr_i<N, double> &dtb);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_COPY_H

