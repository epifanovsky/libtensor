#include <libtensor/block_tensor/btod_trace.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/dag/node_scalar.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_trace.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


class eval_trace_impl {
private:
    enum {
        Nmax = trace::Nmax
    };

private:
    struct dispatch_trace {
        eval_trace_impl &eval;
        expr_tree::node_id_t lhs;
        size_t na;
        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node

public:
    eval_trace_impl(const expr_tree &tr, expr_tree::node_id_t id) :
        m_tree(tr), m_id(id)
    { }

    void evaluate(expr_tree::node_id_t lhs);

    template<size_t N>
    void do_evaluate(expr_tree::node_id_t lhs);

};


void eval_trace_impl::evaluate(expr_tree::node_id_t lhs) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_trace &nt = n.recast_as<node_trace>();

    const node &arga = m_tree.get_vertex(e[0]);
    size_t na = arga.get_n();

    dispatch_trace dt = { *this, lhs, na };
    dispatch_1<2, Nmax>::dispatch(dt, na);
}


template<size_t N>
void eval_trace_impl::do_evaluate(expr_tree::node_id_t lhs) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_trace &nt = n.recast_as<node_trace>();

    enum {
        NA = 2 * N
    };

    btensor_from_node<NA, double> bta(m_tree, e[0]);

    sequence<NA, size_t> seqa1, seqa2;
    for(size_t i = 0; i < N; i++) {
        seqa1[i] = i;
        seqa2[i] = nt.get_idx().at(i);
    }
    for(size_t i = 0; i < N; i++) {
        seqa1[N + i] = N + i;
        seqa2[N + i] = N + nt.get_idx().at(i);
    }
    permutation_builder<NA> pb(seqa1, seqa2);

    permutation<NA> perma(bta.get_transf().get_perm());
    perma.permute(pb.get_perm());

    double d = btod_trace<N>(bta.get_btensor(), perma).calculate();
    d *= bta.get_transf().get_scalar_tr().get_coeff();

    const node_scalar<double> &ns =
        m_tree.get_vertex(lhs).recast_as< node_scalar<double> >();
    ns.get_scalar() = d;
}


template<size_t NA>
void eval_trace_impl::dispatch_trace::dispatch() {

    eval.template do_evaluate<NA/2>(lhs);
}


} // unnamed namespace


void trace::evaluate(node_id_t lhs) {

    eval_trace_impl(m_tree, m_id).evaluate(lhs);
}


//  The code here explicitly instantiates dot_product::evaluate
namespace aux {
template<size_t N>
struct aux_trace {
    trace *e;
    expr_tree::node_id_t lhs;
    aux_trace() { e->evaluate(lhs); }
};
} // namespace aux
template class instantiate_template_1<2, trace::Nmax, aux::aux_trace>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor