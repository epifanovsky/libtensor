#ifndef LABELED_DIRECT_SPARSE_BTENSOR_H
#define LABELED_DIRECT_SPARSE_BTENSOR_H

#include "gen_labeled_btensor.h"
#include "batch_provider_factory.h"

namespace libtensor {

template<size_t N,typename T = double> 
class labeled_direct_sparse_btensor : public gen_labeled_btensor<N,T>
{
private:
    sparse_bispace<N> m_bispace; 
    letter_expr<N> m_le;
    batch_provider<T>** m_batch_provider_ptr_ptr;
public:
    labeled_direct_sparse_btensor(const sparse_bispace<N>& bispace,const letter_expr<N>& le,batch_provider<T>** batch_provider_ptr_ptr) : 
        m_bispace(bispace),m_le(le),m_batch_provider_ptr_ptr(batch_provider_ptr_ptr) {}
    letter_expr<N> get_letter_expr() const { return m_le; }

    const T* get_data_ptr() const { return NULL; }
    virtual batch_provider<T>* get_batch_provider() const { return *m_batch_provider_ptr_ptr; }

    sparse_bispace<N> get_bispace() const { return m_bispace; }; 

    labeled_direct_sparse_btensor<N,T>& operator=(const batch_provider_factory<N,T>& factory);
};

template<size_t N,typename T>
labeled_direct_sparse_btensor<N,T>& labeled_direct_sparse_btensor<N,T>::operator=(const batch_provider_factory<N,T>& factory)
{
    *m_batch_provider_ptr_ptr = factory.get_batch_provider(*this);
    return *this;
}

} // namespace libtensor


#endif /* LABELED_DIRECT_SPARSE_BTENSOR_H */
