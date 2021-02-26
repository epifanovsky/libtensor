#include <libtensor/block_tensor/bto_mult.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "tensor_from_node.h"
#include "eval_btensor_double_div.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N, typename T>
class eval_div_impl : public eval_btensor_evaluator_i<N, T> {
private:
    enum {
        Nmax = div<N, T>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, T>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_div_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, T> &tr);

    virtual ~eval_div_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N, typename T>
eval_div_impl<N, T>::eval_div_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, T> &tr) {

    const expr_tree::edge_list_t &e = tree.get_edges_out(id);

    btensor_from_node<N, T> bta(tree, e[0]);
    btensor_from_node<N, T> btb(tree, e[1]);

    tensor_transf<N, T> tra(bta.get_transf()), trb(btb.get_transf());
    tra.permute(tr.get_perm());
    trb.permute(tr.get_perm());

    m_op = new bto_mult<N, T>(bta.get_btensor(), tra, btb.get_btensor(), trb,
        true, tr.get_scalar_tr());
}


template<size_t N, typename T>
eval_div_impl<N, T>::~eval_div_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N, typename T>
div<N, T>::div(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, T> &tr) :

    m_impl(new eval_div_impl<N, T>(tree, id, tr)) {

}


template<size_t N, typename T>
div<N, T>::~div() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates div<N>
namespace aux {
template<size_t N>
struct aux_div {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, T> *tr;
    const node *t;
    div<N> *e;
    aux_div() {
#pragma noinline
        { e = new div<N>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<T>::Nmax,
    aux::aux_div>;
#endif
template class div<1, double>;
template class div<2, double>;
template class div<3, double>;
template class div<4, double>;
template class div<5, double>;
template class div<6, double>;
template class div<7, double>;
template class div<8, double>;

template class div<1, float>;
template class div<2, float>;
template class div<3, float>;
template class div<4, float>;
template class div<5, float>;
template class div<6, float>;
template class div<7, float>;
template class div<8, float>;


} // namespace eval_btensor_T
} // namespace expr
} // namespace libtensor
