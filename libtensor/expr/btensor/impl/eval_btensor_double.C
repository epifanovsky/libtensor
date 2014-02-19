#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/dag/node_dot_product.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/dag/node_transform.h>
#include "../eval_btensor.h"
#include "metaprog.h"
#include "node_interm.h"
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_dot_product.h"
#include "eval_btensor_double_trace.h"
#include "eval_tree_builder_btensor.h"

#include <libtensor/expr/dag/print_tree.h>

namespace libtensor {
namespace expr {
using namespace eval_btensor_double;


namespace {

class eval_btensor_double_impl {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef eval_tree_builder_btensor::eval_order_t eval_order_t;

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

    void verify_scalar(const node &n);
    void verify_tensor(const node &n);

};


class eval_node {
public:
    static const char k_clazz[]; //!< Class name

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_rhs; //!<  ID of rhs node
    bool m_add; //!< True if evaluate and add

public:
    eval_node(const expr_tree &tr, expr_tree::node_id_t rhs, bool add) :
        m_tree(tr), m_rhs(rhs), m_add(add)
    { }

    void evaluate_scalar(expr_tree::node_id_t lhs);

    template<size_t N>
    void evaluate(const node &lhs);

private:
    /** \brief Gathers information on node
        \param id ID of node to get information about
        \param tr Tensor transformation
        \return ID of operation
     **/
    template<size_t N>
    expr_tree::node_id_t gather_info(expr_tree::node_id_t id,
            tensor_transf<N, double> &tr);
};


const char eval_node::k_clazz[] = "eval_node";


void eval_node::evaluate_scalar(expr_tree::node_id_t lhs) {

    const node &n = m_tree.get_vertex(m_rhs);

    if(n.get_op().compare(node_dot_product::k_op_type) == 0) {
        eval_btensor_double::dot_product(m_tree, m_rhs).evaluate(lhs);
    } else if(n.get_op().compare(node_trace::k_op_type) == 0) {
        eval_btensor_double::trace(m_tree, m_rhs).evaluate(lhs);
    }
}


template<size_t N>
void eval_node::evaluate(const node &lhs) {

    tensor_transf<N, double> tr;
    expr_tree::node_id_t rhs = gather_info<N>(m_rhs, tr);
    const node &n = m_tree.get_vertex(rhs);

    eval_btensor_double::autoselect<N>(m_tree, rhs, tr, m_add).evaluate(lhs);
}


template<size_t N>
expr_tree::node_id_t eval_node::gather_info(
    expr_tree::node_id_t id, tensor_transf<N, double> &tr) {

    const node &n = m_tree.get_vertex(id);
    if (n.get_op().compare(node_transform_base::k_op_type) != 0) {
        return id;
    }

    const node_transform<double> &ntr =
            n.recast_as< node_transform<double> >();

    const std::vector<size_t> &p = ntr.get_perm();
    if(p.size() != N) {
        throw "Bad transform node";
    }
    sequence<N, size_t> s0(0), s1(0);
    for(size_t i = 0; i < N; i++) {
        s0[i] = i;
        s1[i] = p[i];
    }
    permutation_builder<N> pb(s1, s0);
    tr.permute(pb.get_perm());
    tr.transform(ntr.get_coeff());

    return m_tree.get_edges_out(id).front();
}


class eval_assign_tensor {
private:
    const expr_tree &m_tree;
    const node &m_lhs; //!< Left-hand side node (has to be ident or interm)
    expr_tree::node_id_t m_rhs;
    bool m_add; //!< True if addition and assignment

public:
    eval_assign_tensor(const expr_tree &tr, const node &lhs,
        expr_tree::node_id_t rhs, bool add) :
        m_tree(tr), m_lhs(lhs), m_rhs(rhs), m_add(add)
    { }

    template<size_t N>
    void dispatch() {
        eval_node(m_tree, m_rhs, m_add).evaluate<N>(m_lhs);
    }

};


void eval_btensor_double_impl::evaluate() {

    try {

    for (eval_order_t::const_iterator i = m_order.begin();
            i != m_order.end(); i++) {

        const node &n = m_tree.get_vertex(*i);
        if (n.get_op().compare(node_assign::k_op_type) != 0) {
            throw 1;
        }

        handle_assign(*i);
    }

    } catch(int i) {
        std::cout << "exception(int): " << i << std::endl;
        throw;
    } catch(char *p) {
        std::cout << "exception: " << p << std::endl;
        throw;
    }
}


void eval_btensor_double_impl::handle_assign(expr_tree::node_id_t id) {

    const expr_tree::edge_list_t &out = m_tree.get_edges_out(id);
    if(out.size() < 2) {
        throw 11;
    }

    const node &lhs = m_tree.get_vertex(out[0]);

    if(lhs.get_n() > 0) {

        // Check l.h.s.
        verify_tensor(lhs);

        // Evaluate r.h.s. before performing the assignment
        for(size_t i = 1; i < out.size(); i++) {
            eval_assign_tensor e(m_tree, lhs, out[i], (i != 1));
            dispatch_1<1, Nmax>::dispatch(e, lhs.get_n());
        }

        // Put l.h.s. at position of assignment and erase subtree
        m_tree.graph::replace(id, lhs);
        for (size_t i = 0; i < out.size(); i++) m_tree.erase_subtree(out[i]);

    } else {

        // Check l.h.s
        verify_scalar(lhs);

        // Check r.h.s
        if(out.size() != 2) {
            throw 12;
        }

        // Evaluate r.h.s. and assign
        eval_node(m_tree, out[1], false).evaluate_scalar(out[0]);

    }
}


void eval_btensor_double_impl::verify_scalar(const node &t) {

    if(t.get_op().compare(node_scalar_base::k_op_type) == 0) {
        const node_scalar_base &ti = t.recast_as<node_scalar_base>();
        if(ti.get_type() != typeid(double)) {
            throw not_implemented("iface", "eval_btensor", "verify_scalar()",
                __FILE__, __LINE__);
        }
        return;
    }

    throw 2;
}


void eval_btensor_double_impl::verify_tensor(const node &t) {

    if(t.get_op().compare(node_ident::k_op_type) == 0) {
        const node_ident &ti = t.recast_as<node_ident>();
        if(ti.get_type() != typeid(double)) {
            throw not_implemented("iface", "eval_btensor", "verify_tensor()",
                __FILE__, __LINE__);
        }
        return;
    }
    if(t.get_op().compare(node_interm_base::k_op_type) == 0) {
        const node_interm_base &ti = t.recast_as<node_interm_base>();
        if(ti.get_t() != typeid(double)) {
            throw not_implemented("iface", "eval_btensor", "verify_tensor()",
                __FILE__, __LINE__);
        }
        return;
    }

    throw 2;
}


} // unnamed namespace


eval_btensor<double>::~eval_btensor<double>() {

}


bool eval_btensor<double>::can_evaluate(const expr_tree &e) const {

    return true;
}


void eval_btensor<double>::evaluate(const expr_tree &tree) const {

//    std::cout << std::endl;
//    std::cout << "= build plan = " << std::endl;
//    print_tree(tree, std::cout);
    eval_tree_builder_btensor bld(tree);
    bld.build();
//    std::cout << "= process plan =" << std::endl;
//    print_tree(bld.get_tree(), std::cout);
    eval_btensor_double_impl(bld.get_tree(), bld.get_order()).evaluate();
}


} // namespace expr
} // namespace libtensor
