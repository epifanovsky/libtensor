#ifndef LIBTENSOR_TOD_TRACE_H
#define LIBTENSOR_TOD_TRACE_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Computes the trace of a matricized %tensor
    \tparam N Tensor diagonal order.

    This operation computes the sum of the diagonal elements of a matricized
    %tensor:
    \f[
        \textnormal{tr}(A) = \sum_i a_{ii} \qquad
        \textnormal{tr}(B) = \sum_{ij} b_{ijij}
    \f]

    \ingroup libtensor_tod
 **/
template<size_t N, typename T>
class to_trace :
    public timings< to_trace<N, T> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    static const size_t k_ordera = 2 * N; //!< Order of the %tensor

private:
    dense_tensor_rd_i<k_ordera, T> &m_t; //!< Input %tensor
    permutation<k_ordera> m_perm; //!< Permutation of the %tensor

public:
    /** \brief Creates the operation
        \param t Input tensor.
     **/
    to_trace(dense_tensor_rd_i<k_ordera, T> &t);

    /** \brief Creates the operation
        \param t Input tensor.
        \param p Permutation of the tensor.
     **/
    to_trace(dense_tensor_rd_i<k_ordera, T> &t,
        const permutation<k_ordera> &p);

    /** \brief Computes the trace
     **/
    T calculate();

private:
    /** \brief Checks that the dimensions of the input tensor are
            correct or throws an exception
     **/
    void check_dims();

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_TRACE_H
