#include <libtensor/block_tensor/btod_diag.h>
#include <libtensor/expr/dag/node_diag.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_diag.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


class eval_diag_impl {
private:
    enum {
        Nmax = diag::Nmax
    };

private:
    template<size_t N>
    struct dispatch_diag {
        eval_diag_impl &eval;
        const tensor_transf<N, double> &tr;
        const node &t;
        size_t na;
        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_diag_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, const node &t);

    template<size_t N, size_t M>
    void do_evaluate(const tensor_transf<N - M + 1, double> &tr,
        const node &t);

};


template<size_t N>
void eval_diag_impl::evaluate(
    const tensor_transf<N, double> &tr, const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_diag &nd = n.recast_as<node_diag>();

    if (N != t.get_n()) {
        throw "Invalid order";
    }

    const node &arga = m_tree.get_vertex(e[0]);
    size_t na = arga.get_n();

    dispatch_diag<N> dd = { *this, tr, t, na };
    dispatch_1<N + 1, Nmax>::dispatch(dd, na);

//    btensor_i<N, double> &bta = tensor_from_node<N>(m_tree.get_vertex(m_id));
//    btod_copy<N> op(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff());
//    btensor<N, double> &bt =
//            tensor_from_node<N>(t, op.get_bis());
//    if(m_add) {
//        op.perform(bt, 1.0);
//    } else {
//        op.perform(bt);
//    }
}


template<size_t N, size_t M>
void eval_diag_impl::do_evaluate(
    const tensor_transf<N - M + 1, double> &tr, const node &t) {

    enum {
        NA = N,
        NB = N - M + 1
    };

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_diag &nd = n.recast_as<node_diag>();

    btensor_i<NA, double> &bta = tensor_from_node<NA>(m_tree.get_vertex(e[0]));

    mask<NA> m;

    const std::vector<size_t> &idx = nd.get_idx();
    for(size_t i = 0; i < NA; i++) if(idx[i] == nd.get_didx()) m[i] = true;

    btod_diag<N, M> op(bta, m, tr.get_perm(), tr.get_scalar_tr().get_coeff());

    btensor<NB, double> &btb = tensor_from_node<NB>(t, op.get_bis());
    if(m_add) {
        op.perform(btb, 1.0);
    } else {
        op.perform(btb);
    }
}


template<size_t N> template<size_t NA>
void eval_diag_impl::dispatch_diag<N>::dispatch() {

    eval.template do_evaluate<NA, NA + 1 - N>(tr, t);
}


} // unnamed namespace


template<size_t N>
void diag::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_diag_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates diag::evaluate<N>
namespace aux {
template<size_t N>
struct aux_diag {
    diag *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_diag() { e->evaluate(*tr, *n); }
};
} // namespace aux
template class instantiate_template_1<1, diag::Nmax, aux::aux_diag>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor