#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_dirsum.h>
#include <libtensor/expr/dag/node_dirsum.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_dirsum.h"

#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t NC>
class eval_dirsum_impl : public eval_btensor_evaluator_i<NC, double> {
public:
    enum {
        Nmax = dirsum<NC>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<NC, double>::bti_traits
        bti_traits;

private:
    struct dispatch_dirsum {
        eval_dirsum_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t na, nb;
        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    additive_gen_bto<NC, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_dirsum_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<NC, double> &trc);

    virtual ~eval_dirsum_impl();

    virtual additive_gen_bto<NC, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t NA, size_t NB>
    void init(const tensor_transf<NC, double> &trc);

};


template<size_t NC>
eval_dirsum_impl<NC>::eval_dirsum_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<NC, double> &trc) :

    m_tree(tree), m_id(id), m_op(0) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_dirsum &nd = n.recast_as<node_dirsum>();

    const node &arga = m_tree.get_vertex(e[0]);
    const node &argb = m_tree.get_vertex(e[1]);

    size_t na = arga.get_n();
    size_t nb = argb.get_n();

    dispatch_dirsum dd = { *this, trc, na, nb };
    dispatch_1<1, NC - 1>::dispatch(dd, na);
}


template<size_t NC>
eval_dirsum_impl<NC>::~eval_dirsum_impl() {

    delete m_op;
}


template<size_t NC> template<size_t NA, size_t NB>
void eval_dirsum_impl<NC>::init(const tensor_transf<NC, double> &trc) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node_dirsum &nd = m_tree.get_vertex(m_id).recast_as<node_dirsum>();

    btensor_from_node<NA, double> bta(m_tree, e[0]);
    btensor_from_node<NB, double> btb(m_tree, e[1]);

    m_op = new btod_dirsum<NA, NB>(bta.get_btensor(),
        bta.get_transf().get_scalar_tr(), btb.get_btensor(),
        btb.get_transf().get_scalar_tr(), trc);
}


template<size_t NC> template<size_t NA>
void eval_dirsum_impl<NC>::dispatch_dirsum::dispatch() {

    enum {
        NB = NC - NA
    };
    eval.template init<NA, NB>(trc);
}


} // unnamed namespace


template<size_t NC>
dirsum<NC>::dirsum(const expr_tree &tree, node_id_t &id,
    const tensor_transf<NC, double> &tr, bool add) :

    m_impl(new eval_dirsum_impl<NC>(tree, id, tr)), m_add(add) {

}


template<size_t NC>
dirsum<NC>::~dirsum() {

    delete m_impl;
}


template<size_t NC>
void dirsum<NC>::evaluate(const node &t) {

    if(NC != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "dirsum<NC>", "evaluate()",
            "Inconsistent tensor order.");
    }

    additive_gen_bto<NC, bti_traits> &op = m_impl->get_bto();
    btensor<NC, double> &bt = tensor_from_node<NC>(t, op.get_bis());

    if(m_add) {
        gen_block_tensor_rd_ctrl<NC, bti_traits> ctrl(bt);
        std::vector<size_t> nzblk;
        ctrl.req_nonzero_blocks(nzblk);
        addition_schedule<NC, btod_traits> asch(op.get_symmetry(),
            ctrl.req_const_symmetry());
        asch.build(op.get_schedule(), nzblk);

        gen_bto_aux_add<NC, btod_traits> out(op.get_symmetry(), asch, bt,
            scalar_transf<double>());
        out.open();
        op.perform(out);
        out.close();
    } else {
        gen_bto_aux_copy<NC, btod_traits> out(op.get_symmetry(), bt);
        out.open();
        op.perform(out);
        out.close();
    }
}


//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t NC>
struct aux_dirsum {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<NC, double> *tr;
    const node *t;
    dirsum<NC> *e;
    aux_dirsum() { e = new dirsum<NC>(*tree, id, *tr, false); e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_dirsum>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
