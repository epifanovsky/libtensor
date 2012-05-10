#ifndef LIBTENSOR_TOD_SET_CUDA_H
#define LIBTENSOR_TOD_SET_CUDA_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>

namespace libtensor {


/** \brief Sets all elements of a tensor to the given value
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_set_cuda {
private:
    double m_v; //!< Value

public:
    /**	\brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    tod_set_cuda(double v = 0.0);

    /**	\brief Performs the operation
        \param t Output tensor.
     **/
    void perform(dense_tensor_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_CUDA_H
