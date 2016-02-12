#ifndef LIBTENSOR_EXPR_CTF_BTENSOR_H
#define LIBTENSOR_EXPR_CTF_BTENSOR_H

#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/iface/expr_lhs.h>
#include <libtensor/expr/iface/labeled_lhs_rhs.h>
#include <libtensor/expr/bispace/bispace.h>
#include <libtensor/expr/eval/eval.h>
#include "ctf_btensor_i.h"

namespace libtensor {
namespace expr {


/** \brief Distributed CTF block tensor

    \ingroup libtensor_expr_ctf_btensor
 **/
template<size_t N, typename T = double>
class ctf_btensor :
    public ctf_btensor_i<N, T>,
    public expr_lhs<N, T>,
    virtual public ctf_block_tensor<N, T> {

public:
    ctf_btensor(const bispace<N> &bi) :
        ctf_block_tensor<N, T>(bi.get_bis()),
        gen_block_tensor< N, ctf_block_tensor_traits<T> >(bi.get_bis())
    { }

    ctf_btensor(const block_index_space<N> &bis) :
        ctf_block_tensor<N, T>(bis),
        gen_block_tensor< N, ctf_block_tensor_traits<T> >(bis)
    { }

    virtual ~ctf_btensor() { }

    /** \brief Attaches a letter label to ctf_btensor
     **/
    labeled_lhs_rhs<N, T> operator()(const label<N> &label) {
        return labeled_lhs_rhs<N, T>(*this, label,
            any_tensor<N, T>::make_rhs(label));
    }

    /** \brief Computes an expression and assigns it to this tensor
     **/
    virtual void assign(const expr_rhs<N, T> &rhs, const label<N> &label);

    /** \brief Computes an expression and adds it to this tensor
     **/
    virtual void assign_add(const expr_rhs<N, T> &rhs, const label<N> &label);

    /** \brief Converts any_tensor to btensor
     **/
    static ctf_btensor<N, T> &from_any_tensor(any_tensor<N, T> &t);

};


template<size_t N, typename T>
void ctf_btensor<N, T>::assign(const expr_rhs<N, T> &rhs,
    const label<N> &label) {

    node_assign n1(N, false);
    expr_tree e(n1);
    expr_tree::node_id_t id = e.get_root();
    node_ident_any_tensor<N, T> n2(*this);
    e.add(id, n2);

    permutation<N> px = label.permutation_of(rhs.get_label());
    if (! px.is_identity()) {
        std::vector<size_t> perm(N);
        for(size_t i = 0; i < N; i++) perm[i] = px[i];

        node_transform<T> n3(perm, scalar_transf<T>());
        id = e.add(id, n3);
    }
    e.add(id, rhs.get_expr());

    eval().evaluate(e);
}


template<size_t N, typename T>
void ctf_btensor<N, T>::assign_add(const expr_rhs<N, T> &rhs,
    const label<N> &label) {

    node_assign n1(N, true);
    expr_tree e(n1);
    expr_tree::node_id_t id = e.get_root();
    node_ident_any_tensor<N, T> n2(*this);
    e.add(id, n2);

    permutation<N> px = label.permutation_of(rhs.get_label());
    if (! px.is_identity()) {
        std::vector<size_t> perm(N);
        for(size_t i = 0; i < N; i++) perm[i] = px[i];

        node_transform<T> n3(perm, scalar_transf<T>());
        id = e.add(id, n3);
    }
    e.add(id, rhs.get_expr());

    eval().evaluate(e);
}


template<size_t N, typename T>
ctf_btensor<N, T> &ctf_btensor<N, T>::from_any_tensor(any_tensor<N, T> &t) {

    return dynamic_cast< ctf_btensor<N, T>& >(
        t.template get_tensor< ctf_btensor_i<N, T> >());
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::ctf_btensor;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_CTF_BTENSOR_H
