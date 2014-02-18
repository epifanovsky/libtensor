#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_symm.h"

#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_symm_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = symm<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    struct dispatch_symm {
        eval_symm_impl &eval;
        const tensor_transf<N, double> &tr;
        template<size_t M> void dispatch();
    };

    template<size_t M>
    struct tag { };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    additive_gen_bto<N, bti_traits> *m_copyop; //!< Block tensor operation
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_symm_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_symm_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t M>
    void init(const tensor_transf<N, double> &trc, const tag<M>&);

    void init(const tensor_transf<N, double> &trc, const tag<2>&);

    void init(const tensor_transf<N, double> &trc, const tag<3>&);

};


template<size_t N>
eval_symm_impl<N>::eval_symm_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) :

    m_tree(tree), m_id(id), m_op(0), m_copyop(0) {

    const node_symm<double> &n =
        m_tree.get_vertex(m_id).recast_as< node_symm<double> >();

    dispatch_symm d = { *this, tr };
    dispatch_1<2, N>::dispatch(d, n.get_nsym());
}


template<size_t N>
eval_symm_impl<N>::~eval_symm_impl() {

    delete m_op;
    delete m_copyop;
}


template<size_t N>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<2>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

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
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }
    size_t nsymidx = nn.get_sym().size() / 2;
    permutation<N> symperm;
    for(size_t i = 0; i < nsymidx; i++) {
        symperm.permute(nn.get_sym()[2*i], nn.get_sym()[2*i+1]);
    }

    tensor_transf<N, double> tr2(tr), tr2inv(tr2, true);
    tensor_transf<N, double> tpr(bta.get_transf());
    tpr.transform(tr2);
    tensor_transf<N, double> trs(symperm, nn.get_pair_tr());
    tensor_transf<N, double> tspr(tr2inv);
    tspr.transform(trs);
    tspr.transform(tr2);

    m_copyop = new btod_copy<N>(bta.get_btensor(), tpr.get_perm(),
        tpr.get_scalar_tr().get_coeff());
    m_op = new btod_symmetrize2<N>(*m_copyop, tspr.get_perm(),
        tspr.get_scalar_tr().get_coeff() == 1.0);
}


template<size_t N>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<3>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();

    btensor_from_node<N, double> bta(m_tree, e[0]);

    if(nn.get_sym().size() != 3) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }

    m_copyop = new btod_copy<N>(bta.get_btensor(), bta.get_transf().get_perm(),
        bta.get_transf().get_scalar_tr().get_coeff());
    m_op = new btod_symmetrize3<N>(*m_copyop, nn.get_sym().at(0),
        nn.get_sym().at(1), nn.get_sym().at(2),
        nn.get_pair_tr().get_coeff() == 1.0);
}


template<size_t N> template<size_t M>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<M>&) {

    throw not_implemented("libtensor::expr::eval_btensor_double",
        "eval_symm_impl<N>", "init()", __FILE__, __LINE__);
}


template<size_t N> template<size_t M>
void eval_symm_impl<N>::dispatch_symm::dispatch() {

    eval.init(tr, tag<M>());
}


} // unnamed namespace


template<size_t N>
symm<N>::symm(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr, bool add) :

    m_impl(new eval_symm_impl<N>(tree, id, tr)), m_add(add) {

}


template<size_t N>
symm<N>::~symm() {

    delete m_impl;
}


template<size_t N>
void symm<N>::evaluate(const node &t) {

    if(N != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "symm<N>", "evaluate()",
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
struct aux_symm {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    symm<N> *e;
    aux_symm() { e = new symm<N>(*tree, id, *tr, false); e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_symm>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
