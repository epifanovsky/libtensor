#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_copy.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_copy_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = copy<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_copy_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_copy_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N>
eval_copy_impl<N>::eval_copy_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) {

    btensor_from_node<N, double> bta(tree, id);
    m_op = new btod_copy<N>(bta.get_btensor(), tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
}


template<size_t N>
eval_copy_impl<N>::~eval_copy_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N>
copy<N>::copy(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr) :

    m_impl(new eval_copy_impl<N>(tree, id, tr)) {

}


template<size_t N>
copy<N>::~copy() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_copy {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    copy<N> *e;
    aux_copy() {
#pragma noinline
        { e = new copy<N>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_copy>;
#endif
template class copy<1>;
template class copy<2>;
template class copy<3>;
template class copy<4>;
template class copy<5>;
template class copy<6>;
template class copy<7>;
template class copy<8>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
