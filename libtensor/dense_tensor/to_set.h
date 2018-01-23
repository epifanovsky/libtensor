#ifndef LIBTENSOR_TO_SET_H
#define LIBTENSOR_TO_SET_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Changes a tensor by or to a given constant value
    \tparam N Tensor order.

    The operation either adds a given value to a tensor or sets the 
    tensor to the respective value.

    Generalization of tod_set and tof_set

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_set : public timings< to_set<N,T> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    T m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    to_set(T v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
        \param zero Zero tensor first
        \param ta Tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<N, T> &ta);
};

template<size_t N>
using tod_set = to_set<N, double>;

template<size_t N>
using tof_set = to_set<N, float>;
} // namespace libtensor

#endif // LIBTENSOR_TO_SET_H
