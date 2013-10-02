#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/expr/node_ident.h>
#include "../eval_btensor.h"

namespace libtensor {
namespace iface {
using namespace libtensor::expr;


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

template<size_t N> struct tag { };

template<size_t Nmax, typename Tgt>
struct dispatch_1 {
public:
    static void dispatch(Tgt &tgt, size_t n) {
        do_dispatch(tag<Nmax>(), tgt, n);
    }

private:
    template<size_t N>
    static void do_dispatch(const tag<N>&, Tgt &tgt, size_t n) {
        if(N > n) return do_dispatch(tag<N - 1>(), tgt, n);
        if(N == n) tgt.template dispatch<N>();
        else throw "Unable to dispatch";
    }

    static void do_dispatch(const tag<0>&, Tgt &tgt, size_t n) {
        throw "Unable to dispatch";
    }

};

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

class eval_node_ident {
private:
    const tensor_list &m_tl; //!< Tensor list
    const node_ident &m_node; //!< Identity node

public:
    eval_node_ident(const tensor_list &tl, const node_ident &n) :
        m_tl(tl), m_node(n)
    { }

    template<size_t N>
    void evaluate(btensor<N, double> &btb);

};

template<size_t N>
void eval_node::evaluate(btensor<N, double> &bt) {

    if(m_node.get_op().compare("ident") == 0) {
        const node_ident &n = m_node.recast_as<node_ident>();
        eval_node_ident(m_tl, n).evaluate(bt);
    }
}

template<size_t N>
void eval_node_ident::evaluate(btensor<N, double> &btb) {

    btensor_i<N, double> &bta = m_tl.get_tensor<N, double>(m_node.get_tid()).
        template get_tensor< btensor_i<N, double> >();
    btod_copy<N>(bta).perform(btb);
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
    dispatch_1<Nmax, eval_assign>::dispatch(e, tl.get_tensor_order(tid));
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
