#ifndef BATCH_PROVIDER_FACTORY_H
#define BATCH_PROVIDER_FACTORY_H

#include "batch_provider.h"
#include "gen_labeled_btensor.h"

namespace libtensor {

template<size_t N,typename T>
class batch_provider_factory 
{
public:
    virtual batch_provider<T>* get_batch_provider(gen_labeled_btensor<N,T>& output_tensor) const  = 0;
};

} // namespace libtensor

#endif /* BATCH_PROVIDER_FACTORY_H */
