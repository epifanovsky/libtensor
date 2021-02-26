#ifndef LIBTENSOR_TO_SIZE_H
#define LIBTENSOR_TO_SIZE_H

#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Returns the size of a dense tensor
    \tparam N Tensor order.

    This operation returns the size, in bytes, of the memory array that is
    occupied by the tensor. This size includes any alignment or other overhead.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_size : public noncopyable {
public:
    /** \brief Performs the operation
        \param t Tensor.
     **/
    size_t get_size(dense_tensor_rd_i<N, T> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_SIZE_H
