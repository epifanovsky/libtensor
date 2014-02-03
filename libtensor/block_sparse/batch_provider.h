#ifndef BATCH_PROVIDER_H
#define BATCH_PROVIDER_H

#include <map>

namespace libtensor {

template<typename T>
class batch_provider 
{
public:
    virtual void get_batch(T* batch_ptr,const std::map<idx_pair,idx_pair>& output_batches = (std::map<idx_pair,idx_pair>())) = 0;
    virtual ~batch_provider() {}
};

} // namespace libtensor

#endif /* BATCH_PROVIDER_H */
