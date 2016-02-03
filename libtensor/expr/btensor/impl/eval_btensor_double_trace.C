#include <libtensor/block_tensor/btod_trace.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/dag/node_scalar.h>
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

        dispatch_trace(
            eval_trace_impl &eval_,
            expr_tree::node_id_t lhs_,
            size_t na_) :
            eval(eval_), lhs(lhs_), na(na_)
        { }

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

    dispatch_trace dt(*this, lhs, na);
    dispatch_1<2, Nmax>::dispatch(dt, na);
}


template<size_t N>
void eval_trace_impl::do_evaluate(expr_tree::node_id_t lhs) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_trace &nt = n.template recast_as<node_trace>();

    enum {
        NA = 2 * N
    };

    btensor_from_node<NA, double> bta(m_tree, e[0]);

    sequence<NA, size_t> seqa1, seqa2;
    for(size_t i = 0; i < NA; i++) seqa1[i] = i;
    for(size_t k = 0; k < nt.get_cidx().size(); k++) {
        size_t cidx = nt.get_cidx().at(k);
        size_t n = 0;
        for(size_t i = 0; i < NA; i++) if(nt.get_idx().at(i) == cidx) {
            seqa2[i] = n * N + k;
            n++;
        }
    }
    permutation_builder<NA> pb(seqa1, seqa2);

    permutation<NA> perma(bta.get_transf().get_perm());
    perma.permute(pb.get_perm());

    double d = btod_trace<N>(bta.get_btensor(), perma).calculate();
    d *= bta.get_transf().get_scalar_tr().get_coeff();

    const node_scalar<double> &ns =
        m_tree.get_vertex(lhs).template recast_as< node_scalar<double> >();
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


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
