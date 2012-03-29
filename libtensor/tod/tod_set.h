#ifndef LIBTENSOR_TOD_SET_H
#define LIBTENSOR_TOD_SET_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../mp/auto_cpu_lock.h"

namespace libtensor {


/** \brief Sets all elements of a tensor to the given value
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_set {
private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    tod_set(double v = 0.0);

    /** \brief Performs the operation
        \param cpus Pool of CPUs.
        \param t Output tensor.
     **/
    void perform(cpu_pool &cpus, dense_tensor_i<N, double> &t);

};


template<size_t N>
tod_set<N>::tod_set(const double v) :

    m_v(v) {

}


template<size_t N>
void tod_set<N>::perform(cpu_pool &cpus, dense_tensor_i<N, double> &t) {

    dense_tensor_ctrl<N, double> ctrl(t);
    double *d = ctrl.req_dataptr();

    {
        auto_cpu_lock cpu(cpus);
        size_t sz = t.get_dims().get_size();
        for(size_t i = 0; i < sz; i++) d[i] = m_v;
    }

    ctrl.ret_dataptr(d);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_H
