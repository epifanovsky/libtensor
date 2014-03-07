#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_div.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_div_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = div<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_div_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_div_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N>
eval_div_impl<N>::eval_div_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) {

    const expr_tree::edge_list_t &e = tree.get_edges_out(id);

    btensor_from_node<N, double> bta(tree, e[0]);
    btensor_from_node<N, double> btb(tree, e[1]);

    tensor_transf<N, double> tra(bta.get_transf()), trb(btb.get_transf());
    tra.permute(tr.get_perm());
    trb.permute(tr.get_perm());

    m_op = new btod_mult<N>(bta.get_btensor(), tra, btb.get_btensor(), trb,
        true, tr.get_scalar_tr());
}


template<size_t N>
eval_div_impl<N>::~eval_div_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N>
div<N>::div(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr) :

    m_impl(new eval_div_impl<N>(tree, id, tr)) {

}


template<size_t N>
div<N>::~div() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates div<N>
namespace aux {
template<size_t N>
struct aux_div {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    div<N> *e;
    aux_div() {
#pragma noinline
        { e = new div<N>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_div>;
#endif
template class div<1>;
template class div<2>;
template class div<3>;
template class div<4>;
template class div<5>;
template class div<6>;
template class div<7>;
template class div<8>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
