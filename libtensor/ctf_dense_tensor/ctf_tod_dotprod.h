#ifndef LIBTENSOR_CTF_TOD_DOTPROD_H
#define LIBTENSOR_CTF_TOD_DOTPROD_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Computes the dot product of two tensors
    \tparam N Tensor order.

    \sa tod_dotprod

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_dotprod : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_dense_tensor_i<N, double> &m_ta; //!< First tensor (A)
    ctf_dense_tensor_i<N, double> &m_tb; //!< Second tensor (B)
    tensor_transf<N, double> m_tra; //!< Transformation of A
    tensor_transf<N, double> m_trb; //!< Transformation of B

public:
    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tb Second tensor (B)
     **/
    ctf_tod_dotprod(
        ctf_dense_tensor_i<N, double> &ta, ctf_dense_tensor_i<N, double> &tb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param tb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    ctf_tod_dotprod(
        ctf_dense_tensor_i<N, double> &ta, const permutation<N> &perma,
        ctf_dense_tensor_i<N, double> &tb, const permutation<N> &permb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tra Transformation of first tensor (A)
        \param tb Second tensor (B)
        \param trb Transformation of second tensor (B)
     **/
    ctf_tod_dotprod(
        ctf_dense_tensor_i<N, double> &ta, const tensor_transf<N, double> &tra,
        ctf_dense_tensor_i<N, double> &tb, const tensor_transf<N, double> &trb);

    /** \brief Computes the dot product and returns the value
     **/
    double calculate();

private:
    /** \brief Returns true if both arguments have compatible dimensions,
            false otherwise
     **/
    bool verify_dims() const;

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DOTPROD_H
