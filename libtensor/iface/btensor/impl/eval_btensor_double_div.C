#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/expr/dag/node_div.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_div.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_div_impl {
private:
    enum {
        Nmax = div::Nmax
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_div_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

};


template<size_t N>
void eval_div_impl::evaluate(
    const tensor_transf<N, double> &tr, const node &t) {

    if (N != t.get_n()) {
        throw "Invalid order";
    }

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);

    btensor_from_node<N, double> bta(m_tree, e[0]);
    btensor_from_node<N, double> btb(m_tree, e[1]);

    tensor_transf<N, double> tra(bta.get_transf()), trb(btb.get_transf());
    permutation<N> pinvc(tr.get_perm(), true);
    tra.permute(pinvc);
    trb.permute(pinvc);

    btod_mult<N> op(bta.get_btensor(), tra, btb.get_btensor(), trb, true,
        tr.get_scalar_tr());
    btensor<N, double> &btc = tensor_from_node<N>(t, op.get_bis());
    if(m_add) {
        op.perform(btc, 1.0);
    } else {
        op.perform(btc);
    }
}


} // unnamed namespace


template<size_t N>
void div::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_div_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates div::evaluate<N>
namespace aux {
template<size_t N>
struct aux_div {
    div *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_div() { e->evaluate(*tr, *n); }
};
} // namespace aux
template class instantiate_template_1<1, div::Nmax, aux::aux_div>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
