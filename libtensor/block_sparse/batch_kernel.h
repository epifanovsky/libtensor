#ifndef BATCH_KERNEL_H
#define BATCH_KERNEL_H

#include "sparse_defs.h"

namespace libtensor {

template<typename T>
class batch_kernel 
{
public:
    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches) = 0;
    virtual void init(const std::vector<T*>& ptrs,const bispace_batch_map& bbm) {}
};

} // namespace libtensor

#endif /* BATCH_KERNEL_H */
