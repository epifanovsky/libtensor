#ifndef LIBTENSOR_IFACE_BTENSOR_H
#define LIBTENSOR_IFACE_BTENSOR_H

#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/expr/node_assign.h>
#include <libtensor/expr/expr_tree.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform.h>
#include "bispace.h"
#include "btensor_i.h"
#include "expr_lhs.h"
#include "labeled_lhs_rhs.h"
#include "btensor/eval_btensor.h"

namespace libtensor {
namespace iface {


/** \brief User-friendly block tensor

    \ingroup libtensor_iface
 **/
template<size_t N, typename T = double>
class btensor :
    virtual public btensor_i<N, T>,
    virtual public block_tensor< N, T, allocator<T> >,
    public expr_lhs<N, T> {

public:
    btensor(const bispace<N> &bi) :
        block_tensor< N, T, allocator<T> >(bi.get_bis())
    { }

    btensor(const block_index_space<N> &bis) :
        block_tensor< N, T, allocator<T> >(bis)
    { }

    virtual ~btensor() { }

    /** \brief Attaches a letter label to btensor
     **/
    labeled_lhs_rhs<N, T> operator()(const letter_expr<N> &label) {
        return labeled_lhs_rhs<N, T>(*this, label,
            any_tensor<N, T>::make_rhs(label));
    }

    /** \brief Computes an expression and assigns it to this tensor
     **/
    virtual void assign(const expr_rhs<N, T> &rhs, const letter_expr<N> &label);

    /** \brief Converts any_tensor to btensor
     **/
    static btensor<N, T> &from_any_tensor(any_tensor<N, T> &t);

};


template<size_t N, typename T>
void btensor<N, T>::assign(const expr_rhs<N, T> &rhs,
    const letter_expr<N> &label) {

    expr::node_assign n1(N);
    expr::expr_tree e(n1);
    expr::expr_tree::node_id_t id = e.get_root();
    expr::node_ident<N, T> n2(*this);
    e.add(id, n2);

    permutation<N> px = label.permutation_of(rhs.get_label());
    if (! px.is_identity()) {
        std::vector<size_t> perm(N);
        for(size_t i = 0; i < N; i++) perm[i] = px[i];

        expr::node_transform<T> n3(perm, scalar_transf<T>());
        id = e.add(id, n3);
    }
    e.add(id, rhs.get_expr());

    eval_btensor<T>().evaluate(e);
}


template<size_t N, typename T>
btensor<N, T> &btensor<N, T>::from_any_tensor(any_tensor<N, T> &t) {

    return dynamic_cast< btensor<N, T>& >(
        t.template get_tensor< btensor_i<N, T> >());
}


} // namespace iface
} // namespace libtensor


namespace libtensor {

using iface::btensor;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_BTENSOR_H
