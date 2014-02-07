#include <iostream>
#include <set>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/node_ident_any_tensor.h>
#include <libtensor/expr/node_scalar.h>
#include <libtensor/expr/node_transform.h>
#include <libtensor/expr/print_tree.h>
#include "node_interm.h"
#include "metaprog.h"
#include "eval_tree_builder_btensor.h"

namespace libtensor {
namespace iface {
using namespace expr;
using namespace eval_btensor_double;


const char eval_tree_builder_btensor::k_clazz[] = "eval_tree_builder_btensor";


namespace {


/** \brief Insert assignment at current position in tree
 **/
class node_assignment {
public:
    enum {
        Nmax = eval_tree_builder_btensor::Nmax
    };

private:
    expr_tree &m_tree; //!< Evaluation tree
    expr_tree::node_id_t m_cur; //!< ID of current node
    size_t m_n;

public:
    node_assignment(expr_tree &tr, expr_tree::node_id_t cur) :
        m_tree(tr), m_cur(cur), m_n(0) {

        const node &n = m_tree.get_vertex(m_cur);
        m_n = n.get_n();
    }

    void add() {
        dispatch_1<1, Nmax>::dispatch(*this, m_n);
    }

    template<size_t N>
    void dispatch() {

        add<N>();
    }

private:

    template<size_t N>
    void add() {

        const node &n = m_tree.get_vertex(m_cur);
        expr_tree::node_id_t id0 = m_tree.insert(m_cur, node_assign(m_n));
        expr_tree::node_id_t id1 = m_tree.add(id0, node_interm<N, double>());

        // Swap the order of the arguments to id0
        m_tree.erase(id0, m_cur);
        m_tree.graph::add(id0, m_cur);
    }
};


/** \brief Modifies the expression tree for evaluation

    Does some basic validity checks and adds assignments to intermediate
    tensors to the tree.
 **/
class node_renderer {
public:
    static const char k_clazz[];

private:
    expr_tree &m_tree; //!< Evaluation tree
    eval_tree_builder_btensor::eval_order_t &m_order;
    expr_tree::node_id_t m_cur; //!< ID of current node

    bool m_is_ident;
    bool m_is_transf;
    bool m_is_add;

public:
    node_renderer(
        expr_tree &tr, eval_tree_builder_btensor::eval_order_t &order,
        expr_tree::node_id_t cur) :

        m_tree(tr), m_order(order), m_cur(cur),
        m_is_ident(false), m_is_transf(false), m_is_add(false)
    { }

    /** \brief Renders the current subtree
     **/
    void render();

    /** \brief Returns if the current node is a transform node
     **/
    bool is_node_with_transf() const {  return m_is_transf; }

    /** \brief Returns if the current node is an identity or
            intermediate node with possibly being preceded by a transform node
     **/
    bool is_ident() const { return m_is_ident; }

    /** \brief Returns if the current node represents a proper addition
            with possibly being preceded a transform node
     **/
    bool is_addition() const { return m_is_add; }

private:
    //! \brief Renders addition node
    void render_add();

    //! \brief Renders assignment node
    void render_assign();

    //! \brief Renders non-specific node
    void render_gen();

    //! \brief Renders identity node
    void render_ident() {
        m_is_ident = true;
    }

    //! \brief Renders transformation node
    void render_transform();

    /** \brief Replaces addition node n1 if possible
        \return True if successful
     **/
    void replace_addition(expr_tree::node_id_t n1);

    //! \brief Combines several subsequent transforms into one
    void combine_transform(expr_tree::node_id_t id1);

    //! \brief Verify that the given node is a proper l-value
    void verify_lvalue(const node &n);
};


const char node_renderer::k_clazz[] = "node_renderer";


void node_renderer::render()  {

    const node &n = m_tree.get_vertex(m_cur);
    if(n.get_op().compare(node_assign::k_op_type) == 0) {
        render_assign();
    } else if(n.get_op().compare(node_transform_base::k_op_type) == 0) {
        render_transform();
    } else if(n.get_op().compare(node_add::k_op_type) == 0) {
        render_add();
    } else if((n.get_op().compare(node_ident::k_op_type) == 0) ||
            (n.get_op().compare(node_interm_base::k_op_type) == 0)) {
        render_ident();
    } else {
        render_gen();
    }
}


void node_renderer::render_add()  {

    static const char method[] = "render_add()";

    const node &n = m_tree.get_vertex(m_cur);
    const node_add &na = n.recast_as<node_add>();

    const expr_tree::edge_list_t eout(m_tree.get_edges_out(m_cur));

    // Since the order of terms in an addition is arbitrary,
    // merging additions does not preserve the order

    for (size_t i = 0; i != eout.size(); i++) {

        expr_tree::node_id_t id1 = eout[i];
        const node &ni = m_tree.get_vertex(id1);
        if (ni.get_n() != n.get_n()) {
            throw 2;

        }

        node_renderer r(m_tree, m_order, id1);
        r.render();
        if (r.is_ident()) continue;
        if (! r.is_addition()) {
            node_assignment(m_tree, id1).add();

            const expr_tree::edge_list_t &in1 = m_tree.get_edges_in(id1);
            m_order.push_back(in1[0]);
        }
        else replace_addition(id1);
    }

    m_is_add = true;
}


void node_renderer::render_assign()  {

    static const char method[] = "render_assign()";

    expr_tree::edge_list_t out(m_tree.get_edges_out(m_cur));
    if (out.size() < 2) {
        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Invalid assignment.");
    }

    const node &lhs = m_tree.get_vertex(out[0]);
    verify_lvalue(lhs);

    for (size_t i = 1; i < out.size(); i++) {

        const node &rhs = m_tree.get_vertex(out[i]);
        if (rhs.get_n() != lhs.get_n()) {
            throw bad_parameter("iface", k_clazz, method,
                    __FILE__, __LINE__, "Invalid dimensions of lhs and rhs.");
        }

        node_renderer r(m_tree, m_order, out[i]);
        r.render();
        if (r.is_addition()) replace_addition(out[i]);
    }
    m_is_ident = true;
    m_order.push_back(m_cur);
}


void node_renderer::render_gen() {

    const node &n = m_tree.get_vertex(m_cur);
    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_cur);
    for (size_t i = 0; i != e.size(); i++) {

        expr_tree::node_id_t id = e[i];
        node_renderer r(m_tree, m_order, id);
        r.render();
        if (r.is_ident()) continue;
        if (r.is_addition()) replace_addition(id);
        else {
            node_assignment(m_tree, id).add();
            m_order.push_back(e[i]);
        }
    }
}


void node_renderer::render_transform() {

    const expr_tree::edge_list_t &eout = m_tree.get_edges_out(m_cur);
    if (eout.size() != 1) throw 141;

    node_renderer r(m_tree, m_order, eout[0]);
    r.render();
    if (r.is_node_with_transf()) combine_transform(m_cur);

    m_is_transf = true;
    m_is_ident = r.is_ident();
    m_is_add = r.is_addition();
}


void node_renderer::replace_addition(expr_tree::node_id_t id1) {

    const expr_tree::edge_list_t &in = m_tree.get_edges_in(id1);
    if (in.size() != 1) {
        node_assignment(m_tree, id1).add();
        m_order.push_back(in[0]);
    }
    else {
        const node &n0 = m_tree.get_vertex(in[0]);
        if (n0.get_op().compare(node_assign::k_op_type) != 0 &&
            n0.get_op().compare(node_add::k_op_type) != 0) {

            node_assignment(m_tree, id1).add();
            m_order.push_back(in[0]);
        }
    }

    expr_tree::node_id_t id0 = in[0];

    expr_tree::node_id_t id2 = id1;
    const node &n1 = m_tree.get_vertex(id1);
    if (n1.get_op().compare(node_transform_base::k_op_type) == 0) {
        id2 = m_tree.get_edges_out(id1)[0];
    }

    const node &n2 = m_tree.get_vertex(id2);
    if (n2.get_op().compare(node_add::k_op_type) != 0) {
        throw 151;
    }

    expr_tree::edge_list_t out(m_tree.get_edges_out(id2));
    for (size_t i = 0; i < out.size(); i++) {

        m_tree.graph::erase(id2, out[i]);
        m_tree.graph::add(id0, out[i]);
        if (id2 != id1) {
            expr_tree::node_id_t idx =
                    m_tree.insert(out[i], m_tree.get_vertex(id1));
            combine_transform(idx);
        }
    }

    m_tree.erase_subtree(id1);
}


void node_renderer::combine_transform(expr_tree::node_id_t id1)  {

    const node &n1 = m_tree.get_vertex(id1);
    if (n1.get_op().compare(node_transform_base::k_op_type) != 0) return;

    const expr_tree::edge_list_t e1 = m_tree.get_edges_out(id1);
    if (e1.size() != 1) throw 161;

    expr_tree::node_id_t id2 = e1[0];
    const node &n2 = m_tree.get_vertex(id2);
    if (n2.get_op().compare(node_transform_base::k_op_type) != 0) return;

    const expr_tree::edge_list_t e2 = m_tree.get_edges_out(id2);
    if (e2.size() != 1) throw 162;

    const node_transform<double> &ntr1 =
            n1.recast_as< node_transform<double> >();
    const node_transform<double> &ntr2 =
            n2.recast_as< node_transform<double> >();

    const std::vector<size_t> &p1 = ntr1.get_perm(), &p2 = ntr2.get_perm();
    std::vector<size_t> p(p1.size());
    for (size_t i = 0; i < p.size(); i++) p[i] = p2[p1[i]];

    scalar_transf<double> tr(ntr2.get_coeff());
    tr.transform(ntr1.get_coeff());

    m_tree.graph::replace(id1, node_transform<double>(p, tr));
    m_tree.graph::add(id1, e2[0]);
    m_tree.erase(id2);
}


void node_renderer::verify_lvalue(const node &n) {

    static const char method[] = "verify_lvalue(const node &)";

    if (n.get_op().compare(node_ident::k_op_type) == 0) {
        const node_ident &nn = n.recast_as<node_ident>();
        if (nn.get_t() != typeid(double)) {
            throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
        }
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {
        const node_interm_base &nn = n.recast_as<node_interm_base>();
        if (nn.get_t() != typeid(double)) {
            throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
        }
    }
    else if (n.get_op().compare(node_scalar_base::k_op_type) == 0) {
        const node_scalar_base &nn = n.recast_as<node_scalar_base>();
        if (nn.get_type() != typeid(double)) {
            throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
        }
    }
    else {
        throw bad_parameter("iface", k_clazz, method, __FILE__, __LINE__,
            "Invalid lhs.");
    }
}

} // unnamed namespace


void eval_tree_builder_btensor::build() {

    static const char method[] = "build()";

    //  For now assume it's double
    //  TODO: implement other types

    expr_tree::node_id_t head = m_tree.get_root();
    const node &hnode = m_tree.get_vertex(head);
    if (hnode.get_op().compare(node_assign::k_op_type) != 0) {
        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Assignment node missing.");
    }

//    std::cout << "render expression" << std::endl;
//    print_tree(m_tree, std::cout);

    std::set<expr_tree::node_id_t> to_erase;
    for (expr_tree::iterator i = m_tree.begin(); i != m_tree.end(); i++) {

        const node &ni = m_tree.get_vertex(i);
        if (ni.get_op().compare(node_ident::k_op_type) != 0) continue;

        const node_ident &nii = ni.recast_as<node_ident>();

        expr_tree::iterator j = i; j++;
        for (; j != m_tree.end(); j++) {

            const node &nj = m_tree.get_vertex(j);
            if (nj.get_op().compare(node_ident::k_op_type) != 0) continue;

            const node_ident &nij = nj.recast_as<node_ident>();

            if (nii == nij) {
                expr_tree::node_id_t idi = m_tree.get_id(i);
                expr_tree::node_id_t idj = m_tree.get_id(j);
                expr_tree::edge_list_t ej(m_tree.get_edges_in(j));
                for (size_t k = 0; k < ej.size(); k++) {
                    m_tree.graph::replace(ej[k], idj, idi);
                }
                to_erase.insert(idj);
            }
        }
    }
    for (std::set<expr_tree::node_id_t>::iterator i = to_erase.begin();
            i != to_erase.end(); i++) {

        m_tree.erase(*i);
    }

    node_renderer(m_tree, m_order, head).render();

//    std::cout << "rendered expression" << std::endl;
//    print_tree(m_tree, std::cout);

}


} // namespace iface
} // namespace libtensor
