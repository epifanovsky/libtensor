#ifndef GEN_SPARSE_BTENSOR_H
#define GEN_SPARSE_BTENSOR_H

#include "../expr/iface/any_tensor.h"
#include "sparse_bispace.h"

namespace libtensor {

template<typename T>
class batch_provider_i;

template<size_t N,typename T>
class gen_sparse_btensor : public any_tensor<N,T>
{
public:
    virtual const sparse_bispace<N>& get_bispace() const = 0;
    virtual const T* get_data_ptr() const = 0;
    virtual batch_provider_i<T>* get_batch_provider() const = 0;
};

} // namespace libtensor

#endif /* GEN_SPARSE_BTENSOR_H */
