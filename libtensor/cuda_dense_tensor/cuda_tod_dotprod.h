#ifndef LIBTENSOR_CUDA_TOD_DOTPROD_H
#define LIBTENSOR_CUDA_TOD_DOTPROD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>

namespace libtensor {


/** \brief Calculates the inner (dot) product of two tensors
    \tparam N Tensor order.

    \sa tod_dotprod

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N>
class cuda_tod_dotprod :
    public timings< cuda_tod_dotprod<N> >,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< First tensor (A)
    dense_tensor_rd_i<N, double> &m_tb; //!< Second tensor (B)
    permutation<N> m_perma;//!< Permutation of the first tensor (A)
    permutation<N> m_permb; //!< Permutation of the second tensor (B)
    double m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tb Second tensor (B)
     **/
    cuda_tod_dotprod(
        dense_tensor_rd_i<N, double> &ta,
        dense_tensor_rd_i<N, double> &tb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param tb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    cuda_tod_dotprod(
        dense_tensor_rd_i<N, double> &ta, const permutation<N> &perma,
        dense_tensor_rd_i<N, double> &tb, const permutation<N> &permb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tra Transformation of first tensor (A)
        \param tb Second tensor (B)
        \param trb Transformation of second tensor (B)
     **/
    cuda_tod_dotprod(
        dense_tensor_rd_i<N, double> &ta, const tensor_transf<N, double> &tra,
        dense_tensor_rd_i<N, double> &tb, const tensor_transf<N, double> &trb);

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

#endif // LIBTENSOR_CUDA_TOD_DOTPROD_H
