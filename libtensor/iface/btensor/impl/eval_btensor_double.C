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
#include "eval_btensor_double_symm.h"
#include "eval_plan_builder_btensor.h"

namespace libtensor {
namespace iface {
using namespace libtensor::expr;
using namespace eval_btensor_double;


namespace {

class eval_btensor_double_impl {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    const eval_plan &m_plan;
    const tensor_list &m_tl;
    const interm &m_interm;

public:
    eval_btensor_double_impl(const eval_plan &plan, const tensor_list &tl,
        const interm &inter) :
        m_plan(plan), m_tl(tl), m_interm(inter)
    { }

    /** \brief Processes the evaluation plan
     **/
    void evaluate();

private:
    void handle_assign(const expr::node_assign &node);
    void handle_create_interm(tid_t tid);
    void handle_delete_interm(tid_t tid);

    void verify_tensor_type(tid_t tid);

};


class eval_node {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef tensor_list::tid_t tid_t;

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const node &m_node; //!< Expression node
    bool m_add; //!< True if evaluate and add

public:
    eval_node(const tensor_list &tl, const interm &inter, const node &n, bool add) :
        m_tl(tl), m_interm(inter), m_node(n), m_add(add)
    { }

    template<size_t N>
    void evaluate(tid_t tid);

};

const char eval_node::k_clazz[] = "eval_node";

template<size_t N>
void eval_node::evaluate(tid_t tid) {

    node_inspector ni(m_node);

    node_with_transf<N> nwt = ni.template gather_transf<N>();

    if(nwt.n.get_op().compare("ident") == 0) {

        const node_ident &n = nwt.n.template recast_as<node_ident>();
        eval_btensor_double::copy(m_tl, m_interm, n, m_add).
            evaluate(nwt.tr, tid);

    } else if(nwt.n.get_op().compare("contract") == 0) {

        const node_contract &n = nwt.n.template recast_as<node_contract>();
        eval_btensor_double::contract(m_tl, m_interm, n, m_add).
            evaluate(nwt.tr, tid);

    } else if(nwt.n.get_op().compare("symm") == 0) {

        const node_symm<double> &n = nwt.n.template recast_as< node_symm<double> >();
        eval_btensor_double::symm(m_tl, m_interm, n, m_add).
            evaluate(nwt.tr, tid);

    } else {
        throw not_implemented("iface", k_clazz, "evaluate()", __FILE__, __LINE__);
    }
}

class eval_assign {
public:
    typedef tensor_list::tid_t tid_t;

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    tid_t m_tid; //!< Left-hand-side tensor
    const node &m_rhs; //!< Right-hand side of the assignment
    bool m_add; //!< True if addition and assignment

public:
    eval_assign(const tensor_list &tl, const interm &inter, tid_t tid,
        const node &rhs, bool add) :
        m_tl(tl), m_interm(inter), m_tid(tid), m_rhs(rhs), m_add(add)
    { }

    template<size_t N>
    void dispatch() {
        eval_node(m_tl, m_interm, m_rhs, m_add).evaluate<N>(m_tid);
    }

};


void eval_btensor_double_impl::evaluate() {

    try {
    for(eval_plan::iterator i = m_plan.begin(); i != m_plan.end(); ++i) {

        const eval_plan_item &item = m_plan.get_item(i);
        switch(item.code) {
        case eval_plan_action_code::ASSIGN:
            handle_assign(*item.node);
            break;
        case eval_plan_action_code::CREATE_INTERM:
            handle_create_interm(item.tid);
            break;
        case eval_plan_action_code::DELETE_INTERM:
            handle_delete_interm(item.tid);
            break;
        }
    }
    } catch(int i) {
        std::cout << "exception(int): " << i << std::endl;
        throw;
    } catch(char *p) {
        std::cout << "exception: " << p << std::endl;
        throw;
    }
}


void eval_btensor_double_impl::handle_assign(const expr::node_assign &node) {

    tid_t tid = node.get_tid();
    std::cout << "handle_assign " << (void*)tid << std::endl;
    verify_tensor_type(tid);
    eval_assign e(m_tl, m_interm, tid, node.get_rhs(), node.is_add());
    dispatch_1<1, Nmax>::dispatch(e, m_tl.get_tensor_order(tid));
}


void eval_btensor_double_impl::handle_create_interm(tid_t tid) {

    std::cout << "handle_create_interm " << (void*)tid << std::endl;
}


void eval_btensor_double_impl::handle_delete_interm(tid_t tid) {

    std::cout << "handle_delete_interm " << (void*)tid << std::endl;
}


void eval_btensor_double_impl::verify_tensor_type(tid_t tid) {

    if(m_tl.get_tensor_type(tid) != typeid(double)) {
        throw not_implemented("iface", "eval_btensor", "evaluate()", __FILE__, __LINE__);
    }
}


} // unnamed namespace


void eval_btensor<double>::evaluate(expr_tree &tree) {

    std::cout << std::endl;
    std::cout << "= build plan = " << std::endl;
    //const expr::node_assign &na = dynamic_cast<const expr::node_assign&>(tree.get_nodes());
    eval_plan_builder_btensor pbld(tree);
    pbld.build_plan();
    std::cout << "= process plan =" << std::endl;
    eval_btensor_double_impl(pbld.get_plan(), pbld.get_tensors(), pbld.get_interm()).evaluate();
}


} // namespace iface
} // namespace libtensor
