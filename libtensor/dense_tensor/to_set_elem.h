#ifndef LIBTENSOR_TO_SET_ELEM_H
#define LIBTENSOR_TO_SET_ELEM_H

#include <libtensor/core/index.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Assigns a value to a single tensor element
    \tparam N Tensor order.

    This operation allows access to individual tensor elements addressed
    by their index. It is useful to set one or two elements to a particular
    value, but too slow to routinely work with %tensors.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_set_elem : public noncopyable {
public:
    /** \brief Assigns a value to an element of a tensor
        \param t Tensor.
        \param idx Index of element.
        \param d Value.
     **/
    void perform(dense_tensor_wr_i<N, T> &t, const index<N> &idx,
        T d);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_SET_ELEM_H
