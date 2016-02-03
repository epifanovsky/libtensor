#ifndef LIBTENSOR_CTF_TOD_DIRSUM_H
#define LIBTENSOR_CTF_TOD_DIRSUM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Computes the direct sum of two distributed tensors
    \tparam N Order of first tensor.
    \tparam M Order of second tensor.

    \sa tod_dirsum

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M>
class ctf_tod_dirsum : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NB = M,
        NC = N + M
    };

private:
    ctf_dense_tensor_i<NA, double> &m_ta; //!< First tensor (A)
    ctf_dense_tensor_i<NB, double> &m_tb; //!< Second tensor (B)
    double m_ka; //!< Scaling coefficient of A
    double m_kb; //!< Scaling coefficient of B
    double m_c; //!< Scaling coefficient of result (C)
    permutation<NC> m_permc; //!< Permutation of result
    dimensions<NC> m_dimsc; //!< Dimensions of the result

public:
    /** \brief Initializes the operation
        \param ta First input tensor
        \param ka Scalar transformation applied to ta
        \param tb Second input tensor
        \param kb Scalar transformation applied to tb
        \param trc Tensor transformation applied to result
     **/
    ctf_tod_dirsum(
        ctf_dense_tensor_i<NA, double> &ta,
        const scalar_transf<double> &ka,
        ctf_dense_tensor_i<NB, double> &tb,
        const scalar_transf<double> &kb,
        const tensor_transf<NC, double> &trc = tensor_transf<NC, double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_dirsum() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<NC, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DIRSUM_H
