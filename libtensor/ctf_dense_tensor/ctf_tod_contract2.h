#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_H

#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Contracts two distributed tensors
    \tparam N Order of first tensor (A) less contraction degree.
    \tparam M Order of second tensor (B) less contraction degree.
    \tparam K Contraction degree (number of inner indexes).

    \sa tod_contract2

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M, size_t K>
class ctf_tod_contract2 : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    ctf_dense_tensor_i<NA, double> &m_ta; //!< First tensor (A)
    ctf_dense_tensor_i<NB, double> &m_tb; //!< Second tensor (B)
    double m_d; //!< Scaling factor
    to_contract2_dims<N, M, K> m_dimsc; //!< Dimensions of result

public:
    /** \brief Contracts two tensors
        \param contr Contraction.
        \param ta First tensor.
        \param tb Second tensor.
        \param d Scaling coefficient (default 1.0).
     **/
    ctf_tod_contract2(
        const contraction2<N, M, K> &contr,
        ctf_dense_tensor_i<NA, double> &ta,
        ctf_dense_tensor_i<NB, double> &tb,
        double d = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_contract2() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(
        bool zero,
        ctf_dense_tensor_i<NC, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_H
