#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform.h>
#include "../eval_btensor.h"
#include "metaprog.h"
#include "node_inspector.h"
#include "eval_btensor_double_contract.h"
#include "eval_btensor_double_copy.h"

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
public:
    static const char k_clazz[]; //!< Class name

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

const char eval_node::k_clazz[] = "eval_node";

template<size_t N>
void eval_node::evaluate(btensor<N, double> &bt) {

    node_inspector ni(m_node);

    node_with_transf<N> nwt = ni.template gather_transf<N>();

    if(nwt.n.get_op().compare("ident") == 0) {

        const node_ident &n = nwt.n.template recast_as<node_ident>();
        eval_btensor_double::copy(m_tl, n).evaluate(nwt.tr, bt);

    } else if(nwt.n.get_op().compare("contract") == 0) {

        const node_contract &n = nwt.n.template recast_as<node_contract>();
        eval_btensor_double::contract(m_tl, n).evaluate(nwt.tr, bt);

    } else {
        throw not_implemented("iface", k_clazz, "evaluate()", __FILE__, __LINE__);
    }
}

class eval_assign {
public:
    typedef eval_btensor<double>::tid_t tid_t;

private:
    tensor_list &m_tl; //!< Tensor list
    tid_t m_tid; //!< Left-hand-side tensor
    const node &m_rhs; //!< Right-hand side of the assignment

public:
    eval_assign(tensor_list &tl, tid_t tid, const node &rhs) :
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

    tid_t tid = node.get_tid();
    std::cout << "assign " << tid << std::endl;
    verify_tensor_type(tid, tl);
    eval_assign e(tl, tid, node.get_rhs());
    dispatch_1<1, Nmax>::dispatch(e, tl.get_tensor_order(tid));
}


void eval_btensor<double>::handle_create_interm(
    tid_t tid, tensor_list &tl) {

    std::cout << "create_interm " << tid << std::endl;
}


void eval_btensor<double>::handle_delete_interm(
    tid_t tid, tensor_list &tl) {

    std::cout << "delete_interm " << tid << std::endl;
}


void eval_btensor<double>::verify_tensor_type(
    tid_t tid, const tensor_list &tl) {

    if(tl.get_tensor_type(tid) != typeid(double)) {
        throw not_implemented("iface", "eval_btensor", "evaluate()", __FILE__, __LINE__);
    }
}


} // namespace iface
} // namespace libtensor
