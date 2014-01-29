#ifndef DIRECT_SPARSE_BTENSOR_H
#define DIRECT_SPARSE_BTENSOR_H

#include "sparse_bispace.h"

namespace libtensor {

template<size_t N, typename T=double> 
class direct_sparse_btensor
{
private:
    sparse_bispace<N> m_bispace;

public:
    direct_sparse_btensor(const sparse_bispace<N>& bispace) : m_bispace(bispace) {}

};

} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
