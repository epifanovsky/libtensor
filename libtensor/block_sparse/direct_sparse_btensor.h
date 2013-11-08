#ifndef DIRECT_SPARSE_BTENSOR_H
#define DIRECT_SPARSE_BTENSOR_H

#include "lazy_eval_functor.h"

namespace libtensor {

template<size_t N, typename T> 
class direct_sparse_btensor
{
private:
    lazy_eval_functor<N,T>* funct;
public:

};

} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
