#ifndef LIBTENSOR_TOD_SET_H
#define LIBTENSOR_TOD_SET_H

#include <libtensor/timings.h>
#include <libtensor/mp/auto_cpu_lock.h>
#include "dense_tensor_i.h"

namespace libtensor {


/**	\brief Sets all elements of a tensor to a given value
    \tparam N Tensor order.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_set : public timings< tod_set<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    /**	\brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    tod_set(double v = 0.0) : m_v(v) { }

    /**	\brief Performs the operation
        \param cpus Pool of CPUs.
        \param t Output tensor.
     **/
    void perform(cpu_pool &cpus, dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_H
