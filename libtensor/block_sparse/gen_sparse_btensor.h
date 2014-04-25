#ifndef GEN_SPARSE_BTENSOR_H
#define GEN_SPARSE_BTENSOR_H

#include "../expr/iface/any_tensor.h"

namespace libtensor {

template<size_t N,typename T>
class gen_sparse_btensor : public any_tensor<N,T>
{
};

} // namespace libtensor

#endif /* GEN_SPARSE_BTENSOR_H */
