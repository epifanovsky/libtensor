#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/expr/node_ident_any_tensor.h>
#include <libtensor/expr/dag/node_symm.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_symm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_symm_impl {
private:
    enum {
        Nmax = symm::Nmax
    };

private:
    template<size_t N>
    struct dispatch_symm {
        eval_symm_impl &eval;
        const tensor_transf<N, double> &tr;
        const node &t;
        template<size_t M> void dispatch();
    };

    template<size_t M>
    struct tag { };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_symm_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &trc, const node &t);

    template<size_t N, size_t M>
    void do_evaluate(const tensor_transf<N, double> &trc, const node &t,
        const tag<M>&);

    template<size_t N>
    void do_evaluate(const tensor_transf<N, double> &trc, const node &t,
        const tag<2>&);

    template<size_t N>
    void do_evaluate(const tensor_transf<N, double> &trc, const node &t,
        const tag<3>&);

};


template<size_t N>
void eval_symm_impl::evaluate(const tensor_transf<N, double> &tr,
    const node &t) {

    const node_symm<double> &n =
        m_tree.get_vertex(m_id).recast_as< node_symm<double> >();

    dispatch_symm<N> d = { *this, tr, t };
    dispatch_1<2, N>::dispatch(d, n.get_nsym());
}


template<size_t N>
void eval_symm_impl::do_evaluate(const tensor_transf<N, double> &tr,
    const node &t, const tag<2>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if (e.size() != 1) throw "More than one child node";

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();

    // Need to convert
    // T2 S T1 A -> S' T' A, where S = I + Ts and S' = I + Ts'
    //
    // T2 (I + Ts) T1 A =
    // (T2 T1 + T2 Ts T1) A =
    // (T2 T1 + T2 Ts T2(inv) T2 T1) A =
    // [I + T2 Ts T2(inv)] T2 T1 A
    //
    // => Ts' = T2 Ts T2(inv); T' = T2 T1

    btensor_from_node<N, double> bta(m_tree, e[0]);

    if(nn.get_sym().size() % 2 != 0) {
        throw "Wrong size of symmetrization sequence";
    }
    size_t nsymidx = nn.get_sym().size() / 2;
    permutation<N> symperm;
    for(size_t i = 0; i < nsymidx; i++) {
        symperm.permute(nn.get_sym()[i], nn.get_sym()[nsymidx + i]);
    }

    tensor_transf<N, double> tr2(tr), tr2inv(tr2, true);
    tensor_transf<N, double> tpr(bta.get_transf());
    tpr.transform(tr2);
    tensor_transf<N, double> trs(symperm, nn.get_pair_tr());
    tensor_transf<N, double> tspr(tr2inv);
    tspr.transform(trs);
    tspr.transform(tr2);

    btod_copy<N> op(bta.get_btensor(), tpr.get_perm(),
        tpr.get_scalar_tr().get_coeff());
    btod_symmetrize2<N> symop(op, tspr.get_perm(),
        tspr.get_scalar_tr().get_coeff() == 1.0);
    btensor<N, double> &bt = tensor_from_node(t, symop.get_bis());
    if(m_add) {
        symop.perform(bt, 1.0);
    } else {
        symop.perform(bt);
    }
}


template<size_t N>
void eval_symm_impl::do_evaluate(const tensor_transf<N, double> &tr,
    const node &t, const tag<3>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if (e.size() != 1) throw "More than one child node";

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();

    btensor_from_node<N, double> bta(m_tree, e[0]);

    if(nn.get_sym().size() != 3) {
        throw "Wrong size of symmetrization sequence";
    }

    btod_copy<N> op(bta.get_btensor(), bta.get_transf().get_perm(),
        bta.get_transf().get_scalar_tr().get_coeff());
    btod_symmetrize3<N> symop(op, nn.get_sym().at(0), nn.get_sym().at(1),
        nn.get_sym().at(2), nn.get_pair_tr().get_coeff() == 1.0);
    btensor<N, double> &bt = tensor_from_node(t, symop.get_bis());
    if(m_add) {
        symop.perform(bt, 1.0);
    } else {
        symop.perform(bt);
    }
}


template<size_t N, size_t M>
void eval_symm_impl::do_evaluate(const tensor_transf<N, double> &tr,
    const node &t, const tag<M>&) {

    std:: cout << "do_evaluate<" << N << ", " << M << ">" << std::endl;
    throw "High-order symmetrizations not implemented";
}


template<size_t N> template<size_t M>
void eval_symm_impl::dispatch_symm<N>::dispatch() {

    eval.do_evaluate(tr, t, tag<M>());
}


} // unnamed namespace


template<size_t N>
void symm::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_symm_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates symm::evaluate<N>
namespace aux {
template<size_t N>
struct aux_symm {
    symm *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_symm() { e->evaluate(*tr, *n); }
};
} // namespace aux
template class instantiate_template_1<1, symm::Nmax, aux::aux_symm>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
