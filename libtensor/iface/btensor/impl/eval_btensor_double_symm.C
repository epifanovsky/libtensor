#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_symm.h>
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
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_symm_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &trc, const node &t);

    void evaluate(const tensor_transf<1, double> &tr, const node &t);

private:
    template<size_t N>
    expr_tree::node_id_t gather_info(expr_tree::node_id_t id,
        tensor_transf<N, double> &tr);

};


template<size_t N>
void eval_symm_impl::evaluate(const tensor_transf<N, double> &tr,
    const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if (e.size() != 1) throw "More than one child node";

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();
    if(nn.get_nsym() == 2) {

        // Need to convert
        // T2 S T1 A -> S' T' A, where S = I + Ts and S' = I + Ts'
        //
        // T2 (I + Ts) T1 A =
        // (T2 T1 + T2 Ts T1) A =
        // (T2 T1 + T2 Ts T2(inv) T2 T1) A =
        // [I + T2 Ts T2(inv)] T2 T1 A
        //
        // => Ts' = T2 Ts T2(inv); T' = T2 T1

        tensor_transf<N, double> tr1;
        expr_tree::node_id_t id1 = gather_info(e[0], tr1);
        const node &n1 = m_tree.get_vertex(id1);
        btensor_i<N, double> &bta = tensor_from_node<N>(n1);

        if(nn.get_sym().size() % 2 != 0) {
            throw "Wrong size of symmetrization sequence";
        }
        size_t nsymidx = nn.get_sym().size() / 2;
        permutation<N> symperm;
        for(size_t i = 0; i < nsymidx; i++) {
            symperm.permute(nn.get_sym()[i], nn.get_sym()[nsymidx + i]);
        }

        tensor_transf<N, double> tr2(tr), tr2inv(tr2, true);
        tensor_transf<N, double> tpr(tr1);
        tpr.transform(tr2);
        tensor_transf<N, double> trs(symperm, nn.get_pair_tr());
        tensor_transf<N, double> tspr(tr2inv);
        tspr.transform(trs);
        tspr.transform(tr2);

        btod_copy<N> op(bta, tpr.get_perm(), tpr.get_scalar_tr().get_coeff());
        btod_symmetrize2<N> symop(op, tspr.get_perm(), tspr.get_scalar_tr().get_coeff() == 1.0);
        btensor<N, double> &bt = tensor_from_node(t, symop.get_bis());
        if(m_add) {
            symop.perform(bt, 1.0);
        } else {
            symop.perform(bt);
        }

    } else if(nn.get_nsym() == 3) {
        throw "Third-order symmetrizations not implemented";
    } else {
        throw "High-order symmetrizations not implemented";
    }
}


void eval_symm_impl::evaluate(const tensor_transf<1, double> &tr, const node &t) {

    throw "Should not be here";
}


template<size_t N>
expr_tree::node_id_t eval_symm_impl::gather_info(
    expr_tree::node_id_t id, tensor_transf<N, double> &tr) {

    const node &n = m_tree.get_vertex(id);
    if (n.get_op().compare(node_transform_base::k_op_type) != 0) {
        return id;
    }

    const node_transform<double> &ntr =
            n.recast_as< node_transform<double> >();

    const std::vector<size_t> &p = ntr.get_perm();
    if(p.size() != N) {
        throw "Bad transform node";
    }
    sequence<N, size_t> s0(0), s1(0);
    for(size_t i = 0; i < N; i++) {
        s0[i] = i;
        s1[i] = p[i];
    }
    permutation_builder<N> pb(s1, s0);
    tr.permute(pb.get_perm());
    tr.transform(ntr.get_coeff());

    return m_tree.get_edges_out(id).front();
}


} // unnamed namespace


template<size_t N>
void symm::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_symm_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates symm::evaluate<N>
namespace {
template<size_t N>
struct aux_symm {
    symm *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_symm() { e->evaluate(*tr, *n); }
};
} // unnamed namespace
template class instantiate_template_1<1, symm::Nmax, aux_symm>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
