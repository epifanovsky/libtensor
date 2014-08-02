#ifndef DIRECT_SPARSE_BTENSOR_NEW_H
#define DIRECT_SPARSE_BTENSOR_NEW_H

#include "sparse_bispace.h"
#include "labeled_direct_sparse_btensor.h"
#include "gen_sparse_btensor.h"
#include <libtensor/expr/iface/expr_lhs.h>
#include <libtensor/expr/iface/labeled_lhs_rhs.h>

namespace libtensor {

template<size_t N, typename T=double> 
class direct_sparse_btensor_new : public gen_sparse_btensor<N,T>,public expr::expr_lhs<N,T>
{
private:
    sparse_bispace<N> m_bispace;
    batch_provider_i<T>* m_batch_provider;
public:
    direct_sparse_btensor_new(const sparse_bispace<N>& bispace) : m_bispace(bispace),m_batch_provider(NULL) {}
    void set_batch_provider(batch_provider_i<T>& bp) { m_batch_provider = &bp; }

    batch_provider_i<T>* get_batch_provider() const { return m_batch_provider; }

    const sparse_bispace<N>& get_bispace() const { return m_bispace; }
    const T* get_data_ptr() const { return NULL; }

    virtual void assign(const expr::expr_rhs<N, T> &rhs, const expr::label<N> &l) { }

    expr::labeled_lhs_rhs<N, T> operator()(const expr::label<N> &label) {
        return expr::labeled_lhs_rhs<N, T>(*this, label,
            any_tensor<N, T>::make_rhs(label));
    }

    //direct_sparse_btensor_new(const direct_sparse_btensor_new<N,T>& rhs);
    //direct_sparse_btensor_new<N,T>&  operator=(const direct_sparse_btensor_new<N,T>& rhs);
    //~direct_sparse_btensor_new() { if(m_batch_provider != NULL) { delete m_batch_provider; } }
};

} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
