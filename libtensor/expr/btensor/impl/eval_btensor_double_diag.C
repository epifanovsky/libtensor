#include <libtensor/block_tensor/btod_diag.h>
#include <libtensor/expr/dag/node_diag.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_diag.h"

#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_diag_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = diag<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    struct dispatch_diag {
        eval_diag_impl &eval;
        const tensor_transf<N, double> &trc;
        size_t na;
        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_diag_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<N, double> &trc);

    virtual ~eval_diag_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t NA, size_t NB>
    void init(const tensor_transf<N, double> &trc);

};


template<size_t N>
eval_diag_impl<N>::eval_diag_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &trc) :

    m_tree(tree), m_id(id), m_op(0) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node_diag &nd = m_tree.get_vertex(m_id).recast_as<node_diag>();

    const node &arga = m_tree.get_vertex(e[0]);
    size_t na = arga.get_n();

    dispatch_diag dd = { *this, trc, na };
    dispatch_1<N + 1, Nmax>::dispatch(dd, na);
}


template<size_t N>
eval_diag_impl<N>::~eval_diag_impl() {

    delete m_op;
}


template<size_t N> template<size_t NA, size_t M>
void eval_diag_impl<N>::init(const tensor_transf<N, double> &trc) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_diag &nd = n.recast_as<node_diag>();

    btensor_i<NA, double> &bta = tensor_from_node<NA>(m_tree.get_vertex(e[0]));

    mask<NA> m;

    const std::vector<size_t> &idx = nd.get_idx();
    for(size_t i = 0; i < NA; i++) if(idx[i] == nd.get_didx()) m[i] = true;

    m_op = new btod_diag<NA, M>(bta, m, trc.get_perm(),
        trc.get_scalar_tr().get_coeff());
}


template<size_t N> template<size_t NA>
void eval_diag_impl<N>::dispatch_diag::dispatch() {

    eval.template init<NA, NA + 1 - N>(trc);
}


} // unnamed namespace


template<size_t N>
diag<N>::diag(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr, bool add) :

    m_impl(new eval_diag_impl<N>(tree, id, tr)), m_add(add) {

}


template<size_t N>
diag<N>::~diag() {

    delete m_impl;
}


template<size_t N>
void diag<N>::evaluate(const node &t) {

    if(N != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "diag<N>", "evaluate()",
            "Inconsistent tensor order.");
    }

    additive_gen_bto<N, bti_traits> &op = m_impl->get_bto();
    btensor<N, double> &bt = tensor_from_node<N>(t, op.get_bis());

    if(m_add) {
        gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);
        std::vector<size_t> nzblk;
        ctrl.req_nonzero_blocks(nzblk);
        addition_schedule<N, btod_traits> asch(op.get_symmetry(),
            ctrl.req_const_symmetry());
        asch.build(op.get_schedule(), nzblk);

        gen_bto_aux_add<N, btod_traits> out(op.get_symmetry(), asch, bt,
            scalar_transf<double>());
        out.open();
        op.perform(out);
        out.close();
    } else {
        gen_bto_aux_copy<N, btod_traits> out(op.get_symmetry(), bt);
        out.open();
        op.perform(out);
        out.close();
    }
}


//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_diag {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    diag<N> *e;
    aux_diag() { e = new diag<N>(*tree, id, *tr, false); e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_diag>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
