#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_div.h"

#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_div_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = div<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_div_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_div_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N>
eval_div_impl<N>::eval_div_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) {

    const expr_tree::edge_list_t &e = tree.get_edges_out(id);

    btensor_from_node<N, double> bta(tree, e[0]);
    btensor_from_node<N, double> btb(tree, e[1]);

    tensor_transf<N, double> tra(bta.get_transf()), trb(btb.get_transf());
    permutation<N> pinvc(tr.get_perm(), true);
    tra.permute(pinvc);
    trb.permute(pinvc);

    m_op = new btod_mult<N>(bta.get_btensor(), tra, btb.get_btensor(), trb,
        true, tr.get_scalar_tr());
}


template<size_t N>
eval_div_impl<N>::~eval_div_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N>
div<N>::div(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr, bool add) :

    m_impl(new eval_div_impl<N>(tree, id, tr)), m_add(add) {

}


template<size_t N>
div<N>::~div() {

    delete m_impl;
}


template<size_t N>
void div<N>::evaluate(const node &t) {

    if(N != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "div<N>", "evaluate()",
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


//  The code here explicitly instantiates div<N>
namespace aux {
template<size_t N>
struct aux_div {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    div<N> *e;
    aux_div() { e = new div<N>(*tree, id, *tr, false); e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_div>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
