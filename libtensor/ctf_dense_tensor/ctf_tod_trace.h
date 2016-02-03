#ifndef LIBTENSOR_CTF_TOD_TRACE_H
#define LIBTENSOR_CTF_TOD_TRACE_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Computes the trace of a matricized distributed tensor
    \tparam N Tensor diagonal order.

    \sa tod_trace

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_trace : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = 2 * N
    };

private:
    ctf_dense_tensor_i<NA, double> &m_ta; //!< Input tensor
    permutation<NA> m_perma; //!< Permutation of the tensor

public:
    /** \brief Creates the operation
        \param ta Input tensor.
     **/
    ctf_tod_trace(ctf_dense_tensor_i<NA, double> &ta);

    /** \brief Creates the operation
        \param ta Input tensor.
        \param perma Permutation of the tensor.
     **/
    ctf_tod_trace(
        ctf_dense_tensor_i<NA, double> &ta,
        const permutation<NA> &perma);

    /** \brief Computes and returns the trace
     **/
    double calculate();

private:
    /** \brief Checks that the dimensions of the input tensor are
            correct or throws an exception
     **/
    void check_dims();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_TRACE_H
