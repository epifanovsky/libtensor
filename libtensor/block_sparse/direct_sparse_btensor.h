#ifndef DIRECT_SPARSE_BTENSOR_H
#define DIRECT_SPARSE_BTENSOR_H

#include "sparse_bispace.h"
#include "labeled_direct_sparse_btensor.h"

namespace libtensor {

template<size_t N, typename T=double> 
class direct_sparse_btensor
{
private:
    sparse_bispace<N> m_bispace;
    batch_provider<T>* m_batch_provider;
public:
    direct_sparse_btensor(const sparse_bispace<N>& bispace) : m_bispace(bispace),m_batch_provider(NULL) {}
    labeled_direct_sparse_btensor<N,T> operator()(const letter_expr<N>& le);

    void get_batch(T* batch_mem,const std::map<idx_pair,idx_pair>& batches);
    ~direct_sparse_btensor() { if(m_batch_provider != NULL) { delete m_batch_provider; } }
};

template<size_t N,typename T>
labeled_direct_sparse_btensor<N,T> direct_sparse_btensor<N,T>::operator()(const letter_expr<N>& le)
{
    return labeled_direct_sparse_btensor<N,T>(m_bispace,le,&m_batch_provider);
}

template<size_t N,typename T>
void direct_sparse_btensor<N,T>::get_batch(T* batch_mem,const std::map<idx_pair,idx_pair>& batches)
{
    m_batch_provider->get_batch(batch_mem,batches);
}

} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
