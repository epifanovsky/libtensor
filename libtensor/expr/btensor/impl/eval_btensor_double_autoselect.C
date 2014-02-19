#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_diag.h>
#include <libtensor/expr/dag/node_dirsum.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "eval_btensor_double_add.h"
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_contract.h"
#include "eval_btensor_double_copy.h"
#include "eval_btensor_double_diag.h"
#include "eval_btensor_double_dirsum.h"
#include "eval_btensor_double_div.h"
#include "eval_btensor_double_symm.h"
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N>
autoselect<N>::autoselect(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr, bool addd) :

    m_impl(0), m_add(addd) {

    const node &n = tree.get_vertex(id);

    if(n.check_type<node_ident>() || n.check_type<node_interm_base>()) {
        m_impl = new copy<N>(tree, id, tr, addd);
    } else if(n.check_type<node_add>()) {
        m_impl = new add<N>(tree, id);
    } else if(n.check_type<node_contract>()) {
        m_impl = new contract<N>(tree, id, tr, addd);
    } else if(n.check_type<node_diag>()) {
        m_impl = new diag<N>(tree, id, tr, addd);
    } else if(n.check_type<node_dirsum>()) {
        m_impl = new dirsum<N>(tree, id, tr, addd);
    } else if(n.check_type<node_div>()) {
        m_impl = new div<N>(tree, id, tr, addd);
    } else if(n.check_type<node_symm_base>()) {
        m_impl = new symm<N>(tree, id, tr, addd);
    } else {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "autoselect<N>",
            "autoselect()", "Unsupported operation.");
    }
}


template<size_t N>
autoselect<N>::~autoselect() {

    delete m_impl;
}


template<size_t N>
void autoselect<N>::evaluate(const node &t) {

    if(N != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "autoselect<N>",
            "evaluate()", "Inconsistent tensor order.");
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


//  The code here explicitly instantiates autoselect<N>
namespace aux {
template<size_t N>
struct aux_autoselect {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    autoselect<N> *e;
    aux_autoselect() { e = new autoselect<N>(*tree, id, *tr, false);
        e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_autoselect>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
