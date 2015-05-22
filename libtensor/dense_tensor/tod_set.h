#ifndef LIBTENSOR_TOD_SET_H
#define LIBTENSOR_TOD_SET_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Changes a tensor by or to a given constant value
    \tparam N Tensor order.

    The operation either adds a given value to a tensor or sets the 
    tensor to the respective value.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_set : public timings< tod_set<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    tod_set(double v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
        \param zero Zero tensor first
        \param ta Tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<N, double> &ta);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_H
