#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "node_interm.h"
#include "tensor_from_node.h"
#include "eval_btensor_double_copy.h"

#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_copy_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = copy<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_copy_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_copy_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N>
eval_copy_impl<N>::eval_copy_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) {

    btensor_i<N, double> &bta = tensor_from_node<N>(tree.get_vertex(id));
    m_op = new btod_copy<N>(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff());
}


template<size_t N>
eval_copy_impl<N>::~eval_copy_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N>
copy<N>::copy(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr, bool add) :

    m_impl(new eval_copy_impl<N>(tree, id, tr)), m_add(add) {

}


template<size_t N>
copy<N>::~copy() {

    delete m_impl;
}


template<size_t N>
void copy<N>::evaluate(const node &t) {

    if(N != t.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "copy<N>", "evaluate()",
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
struct aux_copy {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    copy<N> *e;
    aux_copy() { e = new copy<N>(*tree, id, *tr, false); e->evaluate(*t); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_copy>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
