#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_dirsum.h>
#include <libtensor/expr/node_dirsum.h>
#include "metaprog.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_dirsum.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_dirsum_impl {
public:
    enum {
        Nmax = dirsum::Nmax
    };

private:
    template<size_t NC>
    struct dispatch_dirsum {
        eval_dirsum_impl &eval;
        const tensor_transf<NC, double> &trc;
        const node &t;
        size_t na, nb;
        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of contraction node
    bool m_add; //!< True if add

public:
    eval_dirsum_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

    template<size_t N, size_t M>
    void do_evaluate(const tensor_transf<N + M, double> &trc, const node &t);

};


template<size_t NC>
void eval_dirsum_impl::evaluate(const tensor_transf<NC, double> &trc,
    const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_dirsum &nd = n.recast_as<node_dirsum>();

    const node &arga = m_tree.get_vertex(e[0]);
    const node &argb = m_tree.get_vertex(e[1]);

    size_t na = arga.get_n();
    size_t nb = argb.get_n();

    dispatch_dirsum<NC> dd = { *this, trc, t, na, nb };
    dispatch_1<1, NC - 1>::dispatch(dd, na);
}


template<size_t N, size_t M>
void eval_dirsum_impl::do_evaluate(
    const tensor_transf<N + M, double> &trc, const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_dirsum &nd = n.recast_as<node_dirsum>();

    btensor_from_node<N, double> bta(m_tree, e[0]);
    btensor_from_node<M, double> btb(m_tree, e[1]);

    btod_dirsum<N, M> op(bta.get_btensor(), bta.get_transf().get_scalar_tr(),
        btb.get_btensor(), btb.get_transf().get_scalar_tr(), trc);
    btensor<N + M, double> &btc = tensor_from_node<N + M>(t, op.get_bis());
    if(m_add) {
        op.perform(btc, 1.0);
    } else {
        op.perform(btc);
    }
}


template<size_t NC> template<size_t NA>
void eval_dirsum_impl::dispatch_dirsum<NC>::dispatch() {

    enum {
        N = NA,
        M = NC - N
    };
    eval.template do_evaluate<N, M>(trc, t);
}


} // unnamed namespace


template<size_t NC>
void dirsum::evaluate(const tensor_transf<NC, double> &trc, const node &t) {

    eval_dirsum_impl(m_tree, m_id, m_add).evaluate(trc, t);
}


//  The code here explicitly instantiates dirsum::evaluate<NC>
namespace {
template<size_t N>
struct aux_dirsum {
    dirsum *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_dirsum() { e->evaluate(*tr, *n); }
};
} // unnamed namespace
template class instantiate_template_1<1, dirsum::Nmax, aux_dirsum>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
