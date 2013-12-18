#include <libtensor/block_tensor/btod_diag.h>
#include <libtensor/expr/node_diag.h>
#include "metaprog.h"
#include "node_interm.h"
#include "eval_btensor_double_diag.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


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

private:
    template<size_t N>
    btensor_i<N, double> &tensor_from_node(const node &n);

    template<size_t N>
    btensor<N, double> &tensor_from_node(const node &n,
        const block_index_space<N> &bis);

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


template<size_t N>
btensor_i<N, double> &eval_diag_impl::tensor_from_node(const node &n) {

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

        return ni.get_tensor().template get_tensor< btensor_i<N, double> >();
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) throw 73;
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


template<size_t N>
btensor<N, double> &eval_diag_impl::tensor_from_node(const node &n,
    const block_index_space<N> &bis) {

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

        return btensor<N, double>::from_any_tensor(ni.get_tensor());
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


} // unnamed namespace


template<size_t N>
void diag::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_diag_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates diag::evaluate<N>
namespace diag_ns {
template<size_t N>
struct aux {
    diag *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux() { e->evaluate(*tr, *n); }
};
} // unnamed namespace
template class instantiate_template_1<1, diag::Nmax, diag_ns::aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
