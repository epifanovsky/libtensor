#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/expr/btensor/btensor_i.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_dot_product.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/expr/eval/tensor_type_check.h>
#include "../eval_ctf_btensor.h"
#include "eval_ctf_btensor_double_autoselect.h"
#include "eval_ctf_btensor_double_convert.h"
#include "eval_ctf_btensor_double_dot_product.h"
#include "eval_ctf_btensor_double_trace.h"
#include "eval_tree_builder_ctf_btensor.h"
#include "ctf_btensor_from_node.h"
#include "node_ctf_btensor_interm.h"

namespace libtensor {
namespace expr {
using namespace eval_ctf_btensor_double;
using eval_btensor_double::dispatch_1;


namespace {

class eval_ctf_btensor_double_impl {
public:
    enum {
        Nmax = eval_ctf_btensor<double>::Nmax
    };

    typedef eval_tree_builder_ctf_btensor::eval_order_t eval_order_t;

private:
    expr_tree &m_tree;
    const eval_order_t &m_order;

public:
    eval_ctf_btensor_double_impl(expr_tree &tr, const eval_order_t &order) :
        m_tree(tr), m_order(order)
    { }

    /** \brief Processes the evaluation plan
     **/
    void evaluate();

private:
    void handle_assign(const expr_tree::node_id_t id);

    void verify_scalar(const node &n);
    void verify_tensor(const node &n);

};


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


const char eval_node::k_clazz[] = "eval_node";


void eval_node::evaluate_scalar(expr_tree::node_id_t lhs) {

    const node &n = m_tree.get_vertex(m_rhs);

    if(n.get_op().compare(node_dot_product::k_op_type) == 0) {
        eval_ctf_btensor_double::dot_product(m_tree, m_rhs).evaluate(lhs);
    } else if(n.get_op().compare(node_trace::k_op_type) == 0) {
        eval_ctf_btensor_double::trace(m_tree, m_rhs).evaluate(lhs);
    }
}


template<size_t N>
void eval_node::evaluate(expr_tree::node_id_t lhs, bool add) {

    tensor_transf<N, double> tr;
    expr_tree::node_id_t rhs = transf_from_node(m_tree, m_rhs, tr);

    //  Check whether we are dealing with the special case of type conversion:
    //  b(i|j) = a(i|j) with different tensor types a and b

    bool type_conversion = false;
    const node &nrhs = m_tree.get_vertex(rhs);
    const node &nlhs = m_tree.get_vertex(lhs);
    if(nlhs.check_type<node_ident>() && nrhs.check_type<node_ident>()) {
        const node_ident_any_tensor<N, double> &nrhs1 =
            nrhs.template recast_as< node_ident_any_tensor<N, double> >();
        const node_ident_any_tensor<N, double> &nlhs1 =
            nlhs.template recast_as< node_ident_any_tensor<N, double> >();
        std::string ttrhs = nrhs1.get_tensor().get_tensor_type();
        std::string ttlhs = nlhs1.get_tensor().get_tensor_type();
        if(ttrhs == ctf_btensor_i<N, double>::k_tensor_type &&
            ttlhs == btensor_i<N, double>::k_tensor_type) {
            type_conversion = true;
        }
        if(ttrhs == btensor_i<N, double>::k_tensor_type &&
            ttlhs == ctf_btensor_i<N, double>::k_tensor_type) {
            type_conversion = true;
        }
    }

    if(type_conversion) {
        eval_ctf_btensor_double::convert<N>(m_tree, rhs).evaluate(lhs);
    } else {
        eval_ctf_btensor_double::autoselect<N>(m_tree, rhs, tr).
            evaluate(lhs, add);
    }
}


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
        eval_node(m_tree, m_rhs).evaluate<N>(m_lhs, m_add);
    }

};


class is_conversion {
private:
    const expr_tree &m_tree;
    expr_tree::node_id_t m_lhs;
    expr_tree::node_id_t m_rhs;
    bool m_valid;

public:
    is_conversion(const expr_tree &tr, expr_tree::node_id_t lhs,
        expr_tree::node_id_t rhs) :
        m_tree(tr), m_lhs(lhs), m_rhs(rhs), m_valid(false)
    { }

    bool valid() const { return m_valid; }

    template<size_t N>
    void dispatch() {
        const node_ident_any_tensor<N, double> &rhs =
            m_tree.get_vertex(m_rhs).
            template recast_as< node_ident_any_tensor<N, double> >();
        const node_ident_any_tensor<N, double> &lhs =
            m_tree.get_vertex(m_lhs).
            template recast_as< node_ident_any_tensor<N, double> >();
        std::string ttrhs = rhs.get_tensor().get_tensor_type();
        std::string ttlhs = lhs.get_tensor().get_tensor_type();
        if(ttrhs == ctf_btensor_i<N, double>::k_tensor_type &&
            ttlhs == btensor_i<N, double>::k_tensor_type) {
            m_valid = true;
        }
        if(ttrhs == btensor_i<N, double>::k_tensor_type &&
            ttlhs == ctf_btensor_i<N, double>::k_tensor_type) {
            m_valid = true;
        }
    }

};


void eval_ctf_btensor_double_impl::evaluate() {

    for (eval_order_t::const_iterator i = m_order.begin();
            i != m_order.end(); i++) {

        const node &n = m_tree.get_vertex(*i);
        if(!n.check_type<node_assign>()) {
            throw eval_exception(__FILE__, __LINE__, "libtensor::expr",
                "eval_ctf_btensor_double_impl", "evaluate()",
                "Evaluator expects an assignment node.");
        }

        handle_assign(*i);
    }
}


void eval_ctf_btensor_double_impl::handle_assign(expr_tree::node_id_t id) {

    const expr_tree::edge_list_t &out = m_tree.get_edges_out(id);
    const node_assign &n = m_tree.get_vertex(id).recast_as<node_assign>();

    if(out.size() != 2) {
        throw eval_exception(__FILE__, __LINE__, "libtensor::expr",
            "eval_ctf_btensor_double_impl", "handle_assign()",
            "Malformed expression (assignment must have two children).");
    }

    const node &lhs = m_tree.get_vertex(out[0]);

    if(lhs.get_n() > 0) {

        // Check l.h.s.
        verify_tensor(lhs);

        // Evaluate r.h.s. before performing the assignment
        eval_assign_tensor e(m_tree, out[0], out[1], n.is_add());
        dispatch_1<1, Nmax>::dispatch(e, lhs.get_n());

        // Put l.h.s. at position of assignment and erase subtree
        m_tree.graph::replace(id, lhs);
        for(size_t i = 0; i < out.size(); i++) m_tree.erase_subtree(out[i]);

    } else {

        // Check l.h.s
        verify_scalar(lhs);

        // Evaluate r.h.s. and assign
        eval_node(m_tree, out[1]).evaluate_scalar(out[0]);

    }
}


void eval_ctf_btensor_double_impl::verify_scalar(const node &t) {

    if(t.check_type<node_scalar_base>()) {
        const node_scalar_base &ti = t.recast_as<node_scalar_base>();
        if(ti.get_type() != typeid(double)) {
            throw not_implemented("libtensor::expr", "eval_btensor_double_impl",
                "verify_scalar()", __FILE__, __LINE__);
        }
        return;
    }

    throw eval_exception(__FILE__, __LINE__,
        "libtensor::expr", "eval_btensor_double_impl", "verify_scalar()",
        "Expect LHS to be a scalar.");
}


void eval_ctf_btensor_double_impl::verify_tensor(const node &t) {

    if(t.check_type<node_ident>()) {
        const node_ident &ti = t.recast_as<node_ident>();
        if(ti.get_type() != typeid(double)) {
            throw not_implemented("libtensor::expr",
                "eval_ctf_btensor_double_impl", "verify_tensor()",
                __FILE__, __LINE__);
        }
        return;
    }
    if(t.check_type<node_ctf_btensor_interm_base>()) {
        const node_ctf_btensor_interm_base &ti =
            t.recast_as<node_ctf_btensor_interm_base>();
        if(ti.get_t() != typeid(double)) {
            throw not_implemented("libtensor::expr",
                "eval_ctf_btensor_double_impl", "verify_tensor()",
                 __FILE__, __LINE__);
        }
        return;
    }

    throw eval_exception(__FILE__, __LINE__,
        "libtensor::expr", "eval_ctf_btensor_double_impl", "verify_tensor()",
        "Expect LHS to be a tensor.");
}


} // unnamed namespace


eval_ctf_btensor<double>::~eval_ctf_btensor<double>() {

}


bool eval_ctf_btensor<double>::can_evaluate(const expr_tree &e) const {

    bool can_eval = tensor_type_check<Nmax, double, ctf_btensor_i>(e);
    if(!can_eval) {
        expr_tree::node_id_t nid = e.get_root();
        if(e.get_vertex(nid).check_type<node_assign>()) {
            const expr_tree::edge_list_t &out = e.get_edges_out(nid);
            const node_assign &n = e.get_vertex(nid).recast_as<node_assign>();
            const node &lhs = e.get_vertex(out[0]);
            const node &rhs = e.get_vertex(out[1]);
            if(lhs.get_n() > 0 && lhs.check_type<node_ident>() &&
                    rhs.check_type<node_ident>()) {
                is_conversion isconv(e, out[0], out[1]);
                dispatch_1<1, Nmax>::dispatch(isconv, lhs.get_n());
                can_eval = isconv.valid();
            }
        }
    }

    return can_eval;
}


void eval_ctf_btensor<double>::evaluate(const expr_tree &tree) const {

    eval_tree_builder_ctf_btensor bld(tree);
    bld.build();

    eval_ctf_btensor_double_impl(bld.get_tree(), bld.get_order()).evaluate();
}


} // namespace expr
} // namespace libtensor
