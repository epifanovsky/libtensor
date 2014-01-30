#ifndef BATCH_PROVIDER_H
#define BATCH_PROVIDER_H

namespace libtensor {

template<typename T>
class batch_provider 
{
public:
    virtual void get_batch(T* batch_ptr) = 0;
    virtual ~batch_provider() {}
};

} // namespace libtensor

#endif /* BATCH_PROVIDER_H */
