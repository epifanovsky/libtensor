#ifndef LIBTENSOR_DIAG_TOD_COPY_H
#define LIBTENSOR_DIAG_TOD_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/permutation.h>
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
class diag_tod_copy : public timings< diag_tod_copy<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_rd_i<N, double> &m_dta; //!< Source tensor A
    permutation<N> m_perma; //!< Permutation for A
    double m_ka; //!< Scaling coefficient for A

public:
    /** \brief Initializes the operation
        \param ta Source tensor A.
     **/
    diag_tod_copy(diag_tensor_rd_i<N, double> &dta) :
        m_dta(dta), m_ka(1.0)
    { }

    /** \brief Initializes the operation
        \param ta Source tensor A.
        \param perma Permutation of A.
        \param ka Scaling factor of A.
     **/
    diag_tod_copy(diag_tensor_rd_i<N, double> &dta, const permutation<N> &perma,
        double ka) :
        m_dta(dta), m_perma(perma), m_ka(ka)
    { }

    /** \brief Performs the operation
        \param zero Zero output before copying.
        \param c Scaling factor.
        \param tb Output tensor.
     **/
    void perform(bool zero, double c, diag_tensor_wr_i<N, double> &tb);

private:
    diag_tod_copy(const diag_tod_copy&);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_COPY_H

