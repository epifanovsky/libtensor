#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_dot_product.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/expr/eval/tensor_type_check.h>
#include "../eval_btensor.h"
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_contract.h"
#include "eval_btensor_double_dot_product.h"
#include "eval_btensor_double_scale.h"
#include "eval_btensor_double_trace.h"
#include "eval_tree_builder_btensor.h"
#include "node_interm.h"
#include "tensor_from_node.h"

namespace libtensor {
namespace expr {
using namespace eval_btensor_double;


namespace {

template<typename T>
class eval_btensor_double_impl {
public:
    enum {
        Nmax = eval_btensor<T>::Nmax
    };

    typedef typename eval_tree_builder_btensor<T>::eval_order_t eval_order_t;

private:
    expr_tree &m_tree;
    const eval_order_t &m_order;

public:
    eval_btensor_double_impl(expr_tree &tr, const eval_order_t &order) :
        m_tree(tr), m_order(order)
    { }

    /** \brief Processes the evaluation plan
     **/
    void evaluate();

private:
    void handle_assign(const expr_tree::node_id_t id);
    void handle_scale(const expr_tree::node_id_t id);

    void verify_scalar(const node &n);
    void verify_tensor(const node &n);

};


template<typename T>
class eval_node {
public:
    static const char k_clazz[]; //!< Class name

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_rhs; //!<  ID of rhs node

public:
    eval_node(const expr_tree &tr, expr_tree::node_id_t rhs) :
        m_tree(tr), m_rhs(rhs)
    { }

    void evaluate_scalar(expr_tree::node_id_t lhs);

    template<size_t N>
    void evaluate(expr_tree::node_id_t lhs, bool add);

};


template<typename T>
const char eval_node<T>::k_clazz[] = "eval_node";

template<typename T>
void eval_node<T>::evaluate_scalar(expr_tree::node_id_t lhs) {

    const node &n = m_tree.get_vertex(m_rhs);

    if(n.get_op().compare(node_dot_product::k_op_type) == 0) {
        eval_btensor_double::dot_product<T>(m_tree, m_rhs).evaluate(lhs);
    } else if(n.get_op().compare(node_trace::k_op_type) == 0) {
        eval_btensor_double::trace<T>(m_tree, m_rhs).evaluate(lhs);
    }
}


template<typename T>
template<size_t N>
void eval_node<T>::evaluate(expr_tree::node_id_t lhs, bool add) {

    tensor_transf<N, T> tr;
    expr_tree::node_id_t rhs = transf_from_node(m_tree, m_rhs, tr);
    const node &n = m_tree.get_vertex(rhs);

    eval_btensor_double::autoselect<N, T>(m_tree, rhs, tr).evaluate(lhs, add);
}

template<typename T>
class eval_assign_tensor {
private:
    const expr_tree &m_tree;
    expr_tree::node_id_t m_lhs; //!< Left-hand side node (has to be ident or interm)
    expr_tree::node_id_t m_rhs;
    bool m_add;

public:
    eval_assign_tensor(const expr_tree &tr, expr_tree::node_id_t lhs,
        expr_tree::node_id_t rhs, bool add) :
        m_tree(tr), m_lhs(lhs), m_rhs(rhs), m_add(add)
    { }

    template<size_t N>
    void dispatch() {
        eval_node<T>(m_tree, m_rhs).template evaluate<N>(m_lhs, m_add);
    }

};

template<typename T>
class eval_scale_tensor {
private:
    const expr_tree &m_tree;
    expr_tree::node_id_t m_lhs;
    expr_tree::node_id_t m_rhs;

public:
    eval_scale_tensor(const expr_tree &tr, expr_tree::node_id_t lhs,
        expr_tree::node_id_t rhs) :
        m_tree(tr), m_lhs(lhs), m_rhs(rhs)
    { }

    template<size_t N>
    void dispatch() {
        eval_btensor_double::scale<N, T>(m_tree, m_rhs).evaluate(m_lhs);
    }

};

template<typename T>
void eval_btensor_double_impl<T>::evaluate() {

    for (typename eval_order_t::const_iterator i = m_order.begin();
            i != m_order.end(); i++) {

        const node &n = m_tree.get_vertex(*i);
        if(n.check_type<node_assign>()) {
            handle_assign(*i);
        } else if(n.check_type<node_scale>()) {
            handle_scale(*i);
        } else {
            throw eval_exception(__FILE__, __LINE__, "libtensor::expr",
                "eval_btensor_double_impl", "evaluate()",
                "Unexpected node type.");
        }
    }
}


template<typename T>
void eval_btensor_double_impl<T>::handle_assign(expr_tree::node_id_t id) {

    const expr_tree::edge_list_t &out = m_tree.get_edges_out(id);
    const node_assign &n = m_tree.get_vertex(id).recast_as<node_assign>();

    if(out.size() != 2) {
        throw eval_exception(__FILE__, __LINE__, "libtensor::expr",
            "eval_btensor_double_impl", "handle_assign()",
            "Malformed expression (assignment must have two children).");
    }

    const node &lhs = m_tree.get_vertex(out[0]);

    if(lhs.get_n() > 0) {

        // Check l.h.s.
        verify_tensor(lhs);

        // Evaluate r.h.s. before performing the assignment
        eval_assign_tensor<T> e(m_tree, out[0], out[1], n.is_add());
        dispatch_1<1, Nmax>::dispatch(e, lhs.get_n());

        // Put l.h.s. at position of assignment and erase subtree
        m_tree.graph::replace(id, lhs);
        for(size_t i = 0; i < out.size(); i++) m_tree.erase_subtree(out[i]);

    } else {

        // Check l.h.s
        verify_scalar(lhs);

        // Evaluate r.h.s. and assign
        eval_node<T>(m_tree, out[1]).evaluate_scalar(out[0]);

    }
}


template<typename T>
void eval_btensor_double_impl<T>::handle_scale(expr_tree::node_id_t id) {

    const expr_tree::edge_list_t &out = m_tree.get_edges_out(id);
    const node_scale &n = m_tree.get_vertex(id).recast_as<node_scale>();

    if(out.size() != 2) {
        throw eval_exception(__FILE__, __LINE__, "libtensor::expr",
            "eval_btensor_double_impl", "handle_scale()",
            "Malformed expression (scaling must have two children).");
    }

    const node &lhs = m_tree.get_vertex(out[0]);

    verify_tensor(lhs);

    eval_scale_tensor<T> e(m_tree, out[0], out[1]);
    dispatch_1<1, Nmax>::dispatch(e, lhs.get_n());
}


template<typename T>
void eval_btensor_double_impl<T>::verify_scalar(const node &t) {

    if(t.check_type<node_scalar_base>()) {
        const node_scalar_base &ti = t.recast_as<node_scalar_base>();
        if(ti.get_type() != typeid(T)) {
            throw not_implemented("libtensor::expr", "eval_btensor_double_impl",
                "verify_scalar()", __FILE__, __LINE__);
        }
        return;
    }

    throw eval_exception(__FILE__, __LINE__,
        "libtensor::expr", "eval_btensor_double_impl", "verify_scalar()",
        "Expect LHS to be a scalar.");
}


template<typename T>
void eval_btensor_double_impl<T>::verify_tensor(const node &t) {

    if(t.check_type<node_ident>()) {
        const node_ident &ti = t.recast_as<node_ident>();
     //       std::cout << "Types:  " << ti.get_type().name() << " vs " << typeid(T).name() << std::endl;
        if(ti.get_type() != typeid(T)) {
            throw not_implemented("libtensor::expr", "eval_btensor_double_impl",
                "verify_tensor()", __FILE__, __LINE__);
        }
        return;
    }
    if(t.check_type<node_interm_base>()) {
        const node_interm_base &ti = t.recast_as<node_interm_base>();
        if(ti.get_t() != typeid(T)) { 
           // std::cout << "Types:  " << ti.get_t().name() << " is not " << typeid(T).name() << std::endl;
            throw not_implemented("libtensor::expr", "eval_btensor_double_impl",
                "verify_tensor()", __FILE__, __LINE__);
    
        }
        return;
    }

    throw eval_exception(__FILE__, __LINE__,
        "libtensor::expr", "eval_btensor_double_impl", "verify_tensor()",
        "Expect LHS to be a tensor.");
}


} // unnamed namespace


template<typename T>
eval_btensor<T>::~eval_btensor<T>() {

}


template<typename T>
bool eval_btensor<T>::can_evaluate(const expr_tree &e) const {

    return tensor_type_check<Nmax, T, btensor_i>(e);
}


template<typename T>
void eval_btensor<T>::evaluate(const expr_tree &tree) const {

    eval_tree_builder_btensor<T> bld(tree);
    bld.build();

    eval_btensor_double_impl<T>(bld.get_tree(), bld.get_order()).evaluate();
}


template<typename T>
void eval_btensor<T>::use_libxm(bool usexm) {

    eval_btensor_double::use_libxm = usexm;
}

template class eval_btensor<double>;
template class eval_btensor<float>;
template class eval_btensor_double_impl<double>;
template class eval_btensor_double_impl<float>;

} // namespace expr
} // namespace libtensor
