#ifndef LIBTENSOR_CTF_TOD_EWMULT2_H
#define LIBTENSOR_CTF_TOD_EWMULT2_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Generalized element-wise multiplication of two tensors
    \tparam N Order of first tensor (A) less the number of shared indices.
    \tparam M Order of second tensor (B) less the number of shared indices.
    \tparam K Number of shared indices.

    \sa tod_ewmult2

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M, size_t K>
class ctf_tod_ewmult2 : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M + K //!< Order of result (C)
    };

private:
    ctf_dense_tensor_i<NA, double> &m_ta; //!< First argument (A)
    tensor_transf<NA, double> m_tra; //!< Transformation of A
    ctf_dense_tensor_i<NB, double> &m_tb; //!< Second argument (B)
    tensor_transf<NB, double> m_trb; //!< Transformation of B
    tensor_transf<NC, double> m_trc; //!< Transformation of result (C)
    dimensions<NC> m_dimsc; //!< Dimensions of result

public:
    /** \brief Initializes the operation
        \param ta First argument (A).
        \param tra Tensor transformation of A.
        \param tb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    ctf_tod_ewmult2(
        ctf_dense_tensor_i<NA, double> &ta,
        const tensor_transf<NA, double> &tra,
        ctf_dense_tensor_i<NB, double> &tb,
        const tensor_transf<NB, double> &trb,
        const tensor_transf<NC, double> &trc = tensor_transf<NC, double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_ewmult2() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(
        bool zero,
        ctf_dense_tensor_i<NC, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_EWMULT2_H
