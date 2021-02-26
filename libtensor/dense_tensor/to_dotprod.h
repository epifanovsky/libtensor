#ifndef LIBTENSOR_TO_DOTPROD_H
#define LIBTENSOR_TO_DOTPROD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Calculates the inner (dot) product of two tensors
    \tparam N Tensor order.

    The inner (dot) product of two tensors is defined as
    \f$ d = sum_{ijk...} a_{ijk...} b_{ijk...} \f$

    Arguments A and B may come with different index order, in which case
    permutations of indexes must be supplied that bring the indexes to
    the same order.


    <b>Examples</b>

    \code
    dense_tensor_i<2, T> &ta = ..., &tb = ...;
    // Compute \sum_{ij} a_{ij} b_{ij}
    T d = to_dotprod<2>(ta, tb).calculate();
    \endcode

    \code
    dense_tensor_i<2, T> &ta = ..., &tb = ...;
    permutation<2> pa, pb;
    pb.permute(0, 1); // ji -> ij
    // Compute \sum_{ij} a_{ij} b_{ji}
    T d = to_dotprod<2>(ta, pa, tb, pb).calculate();
    \endcode

    \code
    dense_tensor_i<3, T> &ta = ..., &tb = ...;
    permutation<3> pa, pb;
    pb.permute(1, 2).permute(0, 1); // jki -> ijk
    // Compute \sum_{ijk} a_{ijk} b_{jki}
    T d = to_dotprod<3>(ta, pa, tb, pb).calculate();
    \endcode

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_dotprod : public timings< to_dotprod<N, T> >, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    dense_tensor_rd_i<N,T> &m_ta; //!< First tensor (A)
    dense_tensor_rd_i<N,T> &m_tb; //!< Second tensor (B)
    permutation<N> m_perma;//!< Permutation of the first tensor (A)
    permutation<N> m_permb; //!< Permutation of the second tensor (B)
    T m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tb Second tensor (B)
     **/
    to_dotprod(dense_tensor_rd_i<N, T> &ta,
        dense_tensor_rd_i<N, T> &tb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param tb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    to_dotprod(dense_tensor_rd_i<N, T> &ta, const permutation<N> &perma,
        dense_tensor_rd_i<N, T> &tb, const permutation<N> &permb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tra Transformation of first tensor (A)
        \param tb Second tensor (B)
        \param trb Transformation of second tensor (B)
     **/
    to_dotprod(
            dense_tensor_rd_i<N, T> &ta,
            const tensor_transf<N, T> &tra,
            dense_tensor_rd_i<N, T> &tb,
            const tensor_transf<N, T> &trb);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Computes the dot product and returns the value
     **/
    T calculate();

private:
    /** \brief Returns true if the dimensions of A and B are compatible
     **/
    bool verify_dims();
};


} // namespace libtensor

#endif // LIBTENSOR_TO_DOTPROD_H
