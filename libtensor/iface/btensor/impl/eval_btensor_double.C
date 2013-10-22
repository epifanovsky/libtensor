#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform_double.h>
#include "../eval_btensor.h"
#include "metaprog.h"
#include "node_inspector.h"
#include "eval_btensor_double_contract.h"

namespace libtensor {
namespace iface {
using namespace libtensor::expr;
using namespace eval_btensor_double;


void eval_btensor<double>::process_plan(
    const eval_plan &plan, tensor_list &tl) {

    for(eval_plan::iterator i = plan.begin(); i != plan.end(); ++i) {

        const eval_plan_item &item = plan.get_item(i);
        switch(item.code) {
        case eval_plan_action_code::ASSIGN:
            handle_assign(*item.node, tl);
            break;
        case eval_plan_action_code::CREATE_INTERM:
            handle_create_interm(item.tid, tl);
            break;
        case eval_plan_action_code::DELETE_INTERM:
            handle_delete_interm(item.tid, tl);
            break;
        }
    }
}


namespace {

class eval_node {
private:
    const tensor_list &m_tl; //!< Tensor list
    const node &m_node; //!< Expression node

public:
    eval_node(const tensor_list &tl, const node &n) :
        m_tl(tl), m_node(n)
    { }

    template<size_t N>
    void evaluate(btensor<N, double> &bt);

};

template<size_t N>
void eval_node::evaluate(btensor<N, double> &bt) {

    node_inspector ni(m_node);

    node_with_transf<N> nwt = ni.template gather_transf<N>();

    if(nwt.n.get_op().compare("ident") == 0) {

        const node_ident &n = nwt.n.template recast_as<node_ident>();
        btensor_i<N, double> &bta = m_tl.get_tensor<N, double>(n.get_tid()).
            template get_tensor< btensor_i<N, double> >();
        btod_copy<N>(bta, nwt.tr.get_perm(),
            nwt.tr.get_scalar_tr().get_coeff()).perform(bt);

    } else if(nwt.n.get_op().compare("contract") == 0) {

        const node_contract &n = nwt.n.template recast_as<node_contract>();
        eval_btensor_double::contract(m_tl, n).evaluate(nwt.tr, bt);

    } else {
        throw "Unknown node type";
    }
}

class eval_assign {
private:
    tensor_list &m_tl; //!< Tensor list
    unsigned m_tid; //!< Left-hand-side tensor
    const node &m_rhs; //!< Right-hand side of the assignment

public:
    eval_assign(tensor_list &tl, unsigned tid, const node &rhs) :
        m_tl(tl), m_tid(tid), m_rhs(rhs)
    { }

    template<size_t N>
    void dispatch() {
        btensor<N, double> &bt = btensor<N, double>::from_any_tensor(
            m_tl.get_tensor<N, double>(m_tid));
        eval_node(m_tl, m_rhs).evaluate(bt);
    }

};

} // unnamed namespace


void eval_btensor<double>::handle_assign(
    const expr::node_assign &node, tensor_list &tl) {

    unsigned tid = node.get_tid();
    verify_tensor_type(tid, tl);
    eval_assign e(tl, tid, node.get_rhs());
    dispatch_1<1, Nmax>::dispatch(e, tl.get_tensor_order(tid));
}


void eval_btensor<double>::handle_create_interm(
    unsigned tid, tensor_list &tl) {

}


void eval_btensor<double>::handle_delete_interm(
    unsigned tid, tensor_list &tl) {

}


void eval_btensor<double>::verify_tensor_type(
    unsigned tid, const tensor_list &tl) {

    if(tl.get_tensor_type(tid) != typeid(double)) {
        throw "Bad tensor type";
    }
}


} // namespace iface
} // namespace libtensor
