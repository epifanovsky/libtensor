#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/ctf_block_tensor/ctf_btod_traits.h>
#include <libtensor/ctf_block_tensor/impl/ctf_btod_set_symmetry.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_diag.h>
#include <libtensor/expr/dag/node_dirsum.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "eval_ctf_btensor_double_add.h"
#include "eval_ctf_btensor_double_autoselect.h"
#include "eval_ctf_btensor_double_contract.h"
#include "eval_ctf_btensor_double_copy.h"
#include "eval_ctf_btensor_double_diag.h"
#include "eval_ctf_btensor_double_dirsum.h"
#include "eval_ctf_btensor_double_div.h"
#include "eval_ctf_btensor_double_set.h"
#include "eval_ctf_btensor_double_symm.h"
#include "ctf_btensor_from_node.h"
#include "node_ctf_btensor_interm.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {


template<size_t N>
autoselect<N>::autoselect(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr) :

    m_tree(tree), m_impl(0) {

    const node &n = m_tree.get_vertex(id);

    if(n.check_type<node_ident>() ||
        n.check_type<node_ctf_btensor_interm_base>()) {
        m_impl = new copy<N>(m_tree, id, tr);
    } else if(n.check_type<node_add>()) {
        m_impl = new add<N>(m_tree, id, tr);
    } else if(n.check_type<node_contract>()) {
        m_impl = new contract<N>(m_tree, id, tr);
    } else if(n.check_type<node_diag>()) {
        m_impl = new diag<N>(m_tree, id, tr);
    } else if(n.check_type<node_dirsum>()) {
        m_impl = new dirsum<N>(m_tree, id, tr);
    } else if(n.check_type<node_div>()) {
        m_impl = new div<N>(m_tree, id, tr);
    } else if(n.check_type<node_set>()) {
        m_impl = new set<N>(m_tree, id, tr);
    } else if(n.check_type<node_symm_base>()) {
        m_impl = new symm<N>(m_tree, id, tr);
    } else {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_ctf_btensor_double", "autoselect<N>",
            "autoselect()", "Unsupported operation.");
    }
}


template<size_t N>
autoselect<N>::~autoselect() {

    delete m_impl;
}


template<size_t N>
void autoselect<N>::evaluate(node_id_t nid_lhs, bool add) {

    const node &lhs = m_tree.get_vertex(nid_lhs);

    if(N != lhs.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_ctf_btensor_double", "autoselect<N>",
            "evaluate()", "Inconsistent tensor order.");
    }

    additive_gen_bto<N, bti_traits> &op = m_impl->get_bto();
    ctf_btensor_from_node<N, double> bt_lhs(m_tree, nid_lhs);
    ctf_btensor<N, double> &bt = bt_lhs.get_or_create_btensor(op.get_bis());

    if(add) {
        gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);
        std::vector<size_t> nzblk;
        ctrl.req_nonzero_blocks(nzblk);
        addition_schedule<N, ctf_btod_traits> asch(op.get_symmetry(),
            ctrl.req_const_symmetry());
        asch.build(op.get_schedule(), nzblk);

        gen_bto_aux_add<N, ctf_btod_traits> out(op.get_symmetry(), asch, bt,
            scalar_transf<double>());
        out.open();
        ctf_btod_set_symmetry<N>().perform(asch, bt);
        op.perform(out);
        out.close();
    } else {
        gen_bto_aux_copy<N, ctf_btod_traits> out(op.get_symmetry(), bt);
        out.open();
        ctf_btod_set_symmetry<N>().perform(op.get_schedule(), bt);
        op.perform(out);
        out.close();
    }
}


#if 0
//  The code here explicitly instantiates autoselect<N>
namespace aux {
template<size_t N>
struct aux_autoselect {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    expr_tree::node_id_t lhs;
    autoselect<N> *e;
    aux_autoselect() {
#pragma noinline
        { e = new autoselect<N>(*tree, id, *tr); e->evaluate(lhs); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_autoselect>;
#endif
template class autoselect<1>;
template class autoselect<2>;
template class autoselect<3>;
template class autoselect<4>;
template class autoselect<5>;
template class autoselect<6>;
template class autoselect<7>;
template class autoselect<8>;


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor
