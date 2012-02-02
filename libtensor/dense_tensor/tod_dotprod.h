#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/mp/cpu_pool.h>
#include "dense_tensor_i.h"

//#include <libtensor/tod/processor.h>

namespace libtensor {


/**	\brief Calculates the inner (dot) product of two tensors
    \tparam N Tensor order.

    The inner (dot) product of two tensors is defined as
    \f$ d = sum_{ijk...} a_{ijk...} b_{ijk...} \f$

    Arguments A and B may come with different index order, in which case
    permutations of indexes must be supplied that bring the indexes to
    the same order.


    <b>Examples</b>

    \code
    dense_tensor_i<2, double> &ta = ..., &tb = ...;
    // Compute \sum_{ij} a_{ij} b_{ij}
    double d = tod_dotprod<2>(ta, tb).calculate();
    \endcode

    \code
    dense_tensor_i<2, double> &ta = ..., &tb = ...;
    permutation<2> pa, pb;
    pb.permute(0, 1); // ji -> ij
    // Compute \sum_{ij} a_{ij} b_{ji}
    double d = tod_dotprod<2>(ta, pa, tb, pb).calculate();
    \endcode

    \code
    dense_tensor_i<3, double> &ta = ..., &tb = ...;
    permutation<3> pa, pb;
    pb.permute(1, 2).permute(0, 1); // jki -> ijk
    // Compute \sum_{ijk} a_{ijk} b_{jki}
    double d = tod_dotprod<3>(ta, pa, tb, pb).calculate();
    \endcode

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_dotprod : public timings< tod_dotprod<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N,double> &m_ta; //!< First tensor (A)
    dense_tensor_rd_i<N,double> &m_tb; //!< Second tensor (B)
    permutation<N> m_perma; //!< Permutation of the first tensor (A)
    permutation<N> m_permb; //!< Permutation of the second tensor (B)

public:
    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tb Second tensor (B)
     **/
    tod_dotprod(dense_tensor_rd_i<N, double> &ta,
        dense_tensor_rd_i<N, double> &tb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param tb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    tod_dotprod(dense_tensor_rd_i<N, double> &ta, const permutation<N> &perma,
        dense_tensor_rd_i<N, double> &tb, const permutation<N> &permb);

    /**	\brief Prefetches the arguments
     **/
    void prefetch();

    /**	\brief Computes the dot product and returns the value
        \param cpus CPU pool.
     **/
    double calculate(cpu_pool &cpus);

private:
    /** \brief Returns true if the dimensions of A and B are compatible
     **/
    bool verify_dims();

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_H
