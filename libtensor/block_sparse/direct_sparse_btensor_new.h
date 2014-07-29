#ifndef DIRECT_SPARSE_BTENSOR_NEW_H
#define DIRECT_SPARSE_BTENSOR_NEW_H

#include "sparse_bispace.h"
#include "labeled_direct_sparse_btensor.h"
#include "gen_sparse_btensor.h"

namespace libtensor {

template<size_t N, typename T=double> 
class direct_sparse_btensor_new : public gen_sparse_btensor<N,T>
{
private:
    sparse_bispace<N> m_bispace;
    batch_provider_i<T>* m_batch_provider;
public:
    direct_sparse_btensor_new(const sparse_bispace<N>& bispace) : m_bispace(bispace),m_batch_provider(NULL) {}
    void set_batch_provider(batch_provider_i<T>& bp) { m_batch_provider = &bp; }
    labeled_direct_sparse_btensor<N,T> operator()(const expr::label<N>& le);

    //void get_batch(T* batch_mem,const std::map<idx_pair,idx_pair>& output_batches,size_t mem_avail = 0);

    void set_batch_provider(const batch_provider<T>& bp);
    batch_provider_i<T>* get_batch_provider() const { return m_batch_provider; }

    const sparse_bispace<N>& get_bispace() const { return m_bispace; }
    const T* get_data_ptr() const { return NULL; }

    //direct_sparse_btensor_new(const direct_sparse_btensor_new<N,T>& rhs);
    //direct_sparse_btensor_new<N,T>&  operator=(const direct_sparse_btensor_new<N,T>& rhs);
    //~direct_sparse_btensor_new() { if(m_batch_provider != NULL) { delete m_batch_provider; } }
};

template<size_t N,typename T>
labeled_direct_sparse_btensor<N,T> direct_sparse_btensor_new<N,T>::operator()(const expr::label<N>& le)
{
    return labeled_direct_sparse_btensor<N,T>(m_bispace,le,&m_batch_provider);
}

//For custom batch providers such as molecular integral interfaces
template<size_t N,typename T>
void direct_sparse_btensor_new<N,T>::set_batch_provider(const batch_provider<T>& bp)
{
    m_batch_provider = bp.clone();
}


#if 0
template<size_t N,typename T>
direct_sparse_btensor_new<N,T>::direct_sparse_btensor_new(const direct_sparse_btensor_new<N,T>& rhs) : m_bispace(rhs.m_bispace),m_batch_provider(NULL)
{
    if(rhs.m_batch_provider != NULL)
    {
        m_batch_provider = rhs.m_batch_provider->clone();
    }
}

template<size_t N,typename T>
direct_sparse_btensor_new<N,T>&  direct_sparse_btensor_new<N,T>::operator=(const direct_sparse_btensor_new<N,T>& rhs)
{
    m_bispace = rhs.m_bispace;
    if(rhs.m_batch_provider != NULL)
    {
        m_batch_provider = rhs.m_batch_provider->clone();
    }
    else
    {
        m_batch_provider = NULL;
    }
}
#endif

#if 0
template<size_t N,typename T>
void direct_sparse_btensor_new<N,T>::get_batch(T* batch_mem,const std::map<idx_pair,idx_pair>& batches,size_t mem_avail)
{
    if(m_batch_provider == NULL)
    {
        throw generic_exception(g_ns,"direct_sparse_btensor_new<N,T>","get_batch(...)",__FILE__,__LINE__,
                "Direct tensor called without being initialized!"); 
    }
    m_batch_provider->get_batch(batch_mem,batches,mem_avail);
}
#endif


} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
