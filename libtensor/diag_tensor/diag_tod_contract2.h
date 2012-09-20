#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Contracts two diagonal tensors
    \param N Order of first tensor less contraction degree.
    \param M Order of second tensor less contraction degree.
    \param K Contraction degree.

    \sa tod_contract2

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_tod_contract2 :
    public timings< diag_tod_contract2<N, M, K> >, public noncopyable {

public:
    static const char *k_clazz; //!< Class name

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    diag_tensor_rd_i<N + K, double> &m_dta; //!< First tensor (A)
    diag_tensor_rd_i<M + K, double> &m_dtb; //!< Second tensor (B)

public:
    /** \brief Initializes the operation
        \param contr Contraction.
        \param dta First tensor (A).
        \param dtb Second tensor (B).
     **/
    diag_tod_contract2(const contraction2<N, M, K> &contr,
        diag_tensor_rd_i<N + K, double> &dta,
        diag_tensor_rd_i<M + K, double> &dtb) :
        m_contr(contr), m_dta(dta), m_dtb(dtb)
    { }

    /** \brief Performs the operation
        \param dtc Output tensor.
     **/
    void perform(diag_tensor_wr_i<N + M, double> &dtc);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_H

