#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/block_tensor/bto_traits.h>
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
#include <libtensor/expr/metaprog.h>
#include "eval_btensor_double_add.h"
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_contract.h"
#include "eval_btensor_double_copy.h"
#include "eval_btensor_double_diag.h"
#include "eval_btensor_double_dirsum.h"
#include "eval_btensor_double_div.h"
#include "eval_btensor_double_set.h"
#include "eval_btensor_double_symm.h"
#include "node_interm.h"
#include "tensor_from_node.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N, typename T>
autoselect<N, T>::autoselect(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, T> &tr) :

    m_tree(tree), m_impl(0) {

    const node &n = m_tree.get_vertex(id);

    if(n.check_type<node_ident>() || n.check_type<node_interm_base>()) {
        m_impl = new copy<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_add>()) {
        m_impl = new add<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_contract>()) {
        m_impl = new contract<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_diag>()) {
        m_impl = new diag<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_dirsum>()) {
        m_impl = new dirsum<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_div>()) {
        m_impl = new div<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_set>()) {
        m_impl = new set<N, T>(m_tree, id, tr);
    } else if(n.check_type<node_symm_base>()) {
        m_impl = new symm<N, T>(m_tree, id, tr);
    } else {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "autoselect<N>",
            "autoselect()", "Unsupported operation.");
    }
}


template<size_t N, typename T>
autoselect<N, T>::~autoselect() {

    delete m_impl;
}


template<size_t N, typename T>
void autoselect<N, T>::evaluate(node_id_t nid_lhs, bool add) {

    const node &lhs = m_tree.get_vertex(nid_lhs);

    if(N != lhs.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "autoselect<N, T>",
            "evaluate()", "Inconsistent tensor order.");
    }

    additive_gen_bto<N, bti_traits> &op = m_impl->get_bto();
    btensor_from_node<N, T> bt_lhs(m_tree, nid_lhs);
    btensor<N, T> &bt = bt_lhs.get_or_create_btensor(op.get_bis());

    if(add) {
        gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);
        std::vector<size_t> nzblk;
        ctrl.req_nonzero_blocks(nzblk);
        addition_schedule<N, bto_traits<T> > asch(op.get_symmetry(),
            ctrl.req_const_symmetry());
        asch.build(op.get_schedule(), nzblk);

        gen_bto_aux_add<N, bto_traits<T> > out(op.get_symmetry(), asch, bt,
            scalar_transf<T>());
        out.open();
        op.perform(out);
        out.close();
    } else {
        gen_bto_aux_copy<N, bto_traits<T> > out(op.get_symmetry(), bt);
        out.open();
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
template class autoselect<1, double>;
template class autoselect<2, double>;
template class autoselect<3, double>;
template class autoselect<4, double>;
template class autoselect<5, double>;
template class autoselect<6, double>;
template class autoselect<7, double>;
template class autoselect<8, double>;

template class autoselect<1, float>;
template class autoselect<2, float>;
template class autoselect<3, float>;
template class autoselect<4, float>;
template class autoselect<5, float>;
template class autoselect<6, float>;
template class autoselect<7, float>;
template class autoselect<8, float>;

} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
