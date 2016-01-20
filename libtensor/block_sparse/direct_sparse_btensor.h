#ifndef DIRECT_SPARSE_BTENSOR_NEW_H
#define DIRECT_SPARSE_BTENSOR_NEW_H

#include "sparse_bispace.h"
#include "gen_sparse_btensor.h"
#include "batch_provider.h"
#include <libtensor/expr/iface/expr_lhs.h>
#include <libtensor/expr/iface/expr_rhs.h>
#include <libtensor/expr/iface/labeled_lhs_rhs.h>
#include <libtensor/expr/dag/node_null.h>

namespace libtensor {

template<size_t N, typename T=double> 
class direct_sparse_btensor : public gen_sparse_btensor<N,T>,public expr::expr_lhs<N,T>
{
private:
    sparse_bispace<N> m_bispace;
    batch_provider_i<T>* m_batch_provider;
    expr::expr_tree* m_expr;
public:
    direct_sparse_btensor(const sparse_bispace<N>& bispace) : m_bispace(bispace),m_batch_provider(NULL),m_expr(new expr::expr_tree(expr::node_null(N))) {}
    void set_batch_provider(batch_provider_i<T>& bp) { m_batch_provider = &bp; }

    batch_provider_i<T>* get_batch_provider() const { return m_batch_provider; }

    const sparse_bispace<N>& get_bispace() const { return m_bispace; }
    const T* get_data_ptr() const { return NULL; }

    virtual void assign(const expr::expr_rhs<N, T> &rhs, const expr::label<N> &l);

    expr::labeled_lhs_rhs<N, T> operator()(const expr::label<N> &lab);

    //STUB!!!! DAT NASTY HACK!!
    virtual void assign_add(const expr::expr_rhs<N, T> &rhs, const expr::label<N> &l)
    {
        throw bad_parameter(g_ns,"direct_sparse_btensor","assign_add",__FILE__, __LINE__,"assign_add not implemented!");
    }
};

template<size_t N,typename T>
void direct_sparse_btensor<N,T>::assign(const expr::expr_rhs<N, T> &rhs, const expr::label<N>& l)
{
    using namespace expr;
    delete m_expr;
    node_assign root(N,false);
    m_expr = new expr_tree(root);
    expr_tree::node_id_t root_id = m_expr->get_root();
    node_ident_any_tensor<N,T> n_tensor(*this);
    m_expr->add(root_id,n_tensor);

    permutation<N> perm = l.permutation_of(rhs.get_label());
    if(!perm.is_identity()) 
    {
        std::vector<size_t> perm_entries(N);
        for(size_t i = 0; i < N; i++) perm_entries[i] = perm[i];

        node_transform<T> n_tf(perm_entries, scalar_transf<T>());
        root_id = m_expr->add(root_id,n_tf);
    }
    m_expr->add(root_id, rhs.get_expr());
}

template<size_t N,typename T>
expr::labeled_lhs_rhs<N, T> direct_sparse_btensor<N,T>::operator()(const expr::label<N> &lab)
{
    if(m_batch_provider != NULL)
    {
        return expr::labeled_lhs_rhs<N, T>(*this, lab,any_tensor<N, T>::make_rhs(lab));
    }
    else
    {
        return expr::labeled_lhs_rhs<N, T>(*this, lab, expr::expr_rhs<N, T>(*m_expr, lab));
    }
}

} // namespace libtensor



#endif /* DIRECT_SPARSE_BTENSOR_H */
