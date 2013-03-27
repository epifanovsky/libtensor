#ifndef LIBTENSOR_CUDA_TOD_SET_H
#define LIBTENSOR_CUDA_TOD_SET_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>

namespace libtensor {


/** \brief Sets all elements of a tensor to the given value
    \tparam N Tensor order.

    \ingroup libtensor_cuda_tod
 **/
template<size_t N>
class cuda_tod_set : public noncopyable {
private:
    double m_v; //!< Value

public:
    static const char *k_clazz; //!< Class name

    /**	\brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    cuda_tod_set(double v = 0.0);

    /**	\brief Performs the operation
        \param t Output tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_SET_H
