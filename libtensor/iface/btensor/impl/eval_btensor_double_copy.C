#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/expr/node_ident.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_copy.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_copy_impl {
private:
    enum {
        Nmax = copy::Nmax
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_copy_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

};


template<size_t N>
void eval_copy_impl::evaluate(
    const tensor_transf<N, double> &tr, const node &t) {

    if (N != t.get_n()) {
        throw "Invalid order";
    }

    btensor_i<N, double> &bta = tensor_from_node<N>(m_tree.get_vertex(m_id));
    btod_copy<N> op(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff());
    btensor<N, double> &bt =
            tensor_from_node<N>(t, op.get_bis());
    if(m_add) {
        op.perform(bt, 1.0);
    } else {
        op.perform(bt);
    }
}


} // unnamed namespace


template<size_t N>
void copy::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_copy_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates copy::evaluate<N>
namespace aux {
template<size_t N>
struct aux_copy {
    copy *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_copy() { e->evaluate(*tr, *n); }
};
} // namespace aux
template class instantiate_template_1<1, copy::Nmax, aux::aux_copy>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
