#include <iostream>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_assign.h>
#include <libtensor/expr/node_ident.h>
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

        m_tree(tr), m_cur(cur), m_n(0)
    {
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
    expr_tree::node_id_t m_cur; //!< ID of current node
    bool m_is_ident;
    bool m_is_transf;
    bool m_is_add;

public:
    node_renderer(expr_tree &tr, expr_tree::node_id_t cur) :

        m_tree(tr), m_cur(cur),
        m_is_ident(false), m_is_transf(false), m_is_add(false)
    { }

    void render() {

        const node &n = m_tree.get_vertex(m_cur);
        if(n.get_op().compare(node_assign::k_op_type) == 0) {
            render_assign();
        } else if(n.get_op().compare(node_transform_base::k_op_type) == 0) {
            render_transform();
        } else if(n.get_op().compare(node_add::k_op_type) == 0) {
            render_add();
        } else if((n.get_op().compare(node_ident_base::k_op_type) == 0) ||
                (n.get_op().compare(node_interm_base::k_op_type) == 0)) {
            render_ident();
        } else {
            render_op();
        }
    }

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
    void render_transform() {

        const expr_tree::edge_list_t &eout = m_tree.get_edges_out(m_cur);
        if (eout.size() != 1) {
            throw 1;
        }

        node_renderer r(m_tree, eout[0]);
        r.render();
        if (r.is_node_with_transf()) combine_transform(m_cur);
        m_is_transf = true;
        m_is_ident = r.is_ident();
        m_is_add = r.is_addition();
    }

    void render_add() {

        static const char method[] = "render_add()";

        const node &n = m_tree.get_vertex(m_cur);
        const node_add &na = n.recast_as<node_add>();

        const expr_tree::edge_list_t &eout = m_tree.get_edges_out(m_cur);

        // Since the order of terms in an addition is arbitrary,
        // merging additions does not preserve the order

        for (size_t i = 0; i != eout.size(); i++) {

            expr_tree::node_id_t id1 = eout[i];
            const node &ni = m_tree.get_vertex(id1);
            if (ni.get_n() != n.get_n()) {
                throw 2;
            }

            node_renderer r(m_tree, id1);
            r.render();
            if (r.is_ident()) continue;
            if (! r.is_addition()) {
                node_assignment(m_tree, id1).add();
            }

            if (r.is_node_with_transf()) {

                const node &ntr = m_tree.get_vertex(id1);
                const expr_tree::edge_list_t &e1 = m_tree.get_edges_out(id1);

                expr_tree::node_id_t id2 = e1[0];
                expr_tree::edge_list_t e2(m_tree.get_edges_out(id2));
                for (size_t i = 0; i < e2.size(); i++) {
                    expr_tree::node_id_t cur = m_tree.add(m_cur, ntr);
                    m_tree.graph::add(cur, e2[i]);
                    m_tree.erase(id2, e2[i]);

                    const std::string &op = m_tree.get_vertex(e2[i]).get_op();
                    if (op.compare(node_transform_base::k_op_type) == 0) {
                        combine_transform(cur);
                    }
                }
            }
            else {
                expr_tree::edge_list_t e1(m_tree.get_edges_out(id1));
                for (size_t i = 0; i < e1.size(); i++) {
                    m_tree.graph::add(m_cur, e1[i]);
                    m_tree.erase(id1, e1[i]);
                }
            }

            m_tree.erase(id1);
        }

        m_is_add = true;
    }

    void render_ident() {

        m_is_ident = true;
    }

    void render_assign() {

        static const char method[] = "render_assign()";

        const node &n = m_tree.get_vertex(m_cur);

        const expr_tree::edge_list_t &out = m_tree.get_edges_out(m_cur);
        if (out.size() != 2) {
            throw bad_parameter("iface", k_clazz, method,
                    __FILE__, __LINE__, "Invalid assignment.");
        }

        const node &lhs = m_tree.get_vertex(out[0]);
        const node &rhs = m_tree.get_vertex(out[1]);
        if (lhs.get_n() != rhs.get_n()) {
            throw bad_parameter("iface", k_clazz, method,
                    __FILE__, __LINE__, "Invalid dimensions of lhs and rhs.");
        }

        if (lhs.get_op().compare(node_ident_base::k_op_type) == 0) {
            const node_ident_base &n =
                    lhs.recast_as<node_ident_base>();
            if (n.get_t() != typeid(double)) {
                throw not_implemented("iface", k_clazz,
                        method, __FILE__, __LINE__);
            }
        }
        else if (lhs.get_op().compare(node_interm_base::k_op_type) == 0) {
            const node_interm_base &n =
                    lhs.recast_as<node_interm_base>();
            if (n.get_t() != typeid(double)) {
                throw not_implemented("iface", k_clazz,
                        method, __FILE__, __LINE__);
            }
        }
        else {
            throw bad_parameter("iface", k_clazz, method,
                    __FILE__, __LINE__, "Invalid lhs.");
        }

        node_renderer(m_tree, out[0]).render();
        node_renderer(m_tree, out[1]).render();

        m_is_ident = true;
    }

    void render_op() {

        static const char method[] = "render_op()";

        const node &n = m_tree.get_vertex(m_cur);
        const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_cur);
        for (size_t i = 0; i != e.size(); i++) {

            expr_tree::node_id_t id = e[i];
            node_renderer r(m_tree, id);
            r.render();
            if (r.is_ident()) continue;

            node_assignment(m_tree, e[i]).add();
        }
    }

    void combine_transform(expr_tree::node_id_t id1) {

        const expr_tree::edge_list_t &e1 = m_tree.get_edges_out(id1);
        expr_tree::node_id_t id2 = e1[0];

        const node &n1 = m_tree.get_vertex(id1);
        const node &n2 = m_tree.get_vertex(id2);

        if (n1.get_op().compare(node_transform_base::k_op_type) != 0) {
            throw 3;
        }
        if (n1.get_op().compare(node_transform_base::k_op_type) != 0) {
            throw 4;
        }

        const node_transform<double> &tr1 =
                n1.recast_as< node_transform<double> >();
        const node_transform<double> &tr2 =
                n2.recast_as< node_transform<double> >();

        const std::vector<size_t> &p1 = tr1.get_perm(), &p2  = tr2.get_perm();
        std::vector<size_t> p(p1.size());
        for (size_t i = 0; i < p.size(); i++) p[i] = p2[p1[i]];

        scalar_transf<double> str(tr2.get_coeff());
        str.transform(tr1.get_coeff());

        expr_tree::node_id_t id0 =
                m_tree.insert(id1, node_transform<double>(p, str));

        const expr_tree::edge_list_t &e2 = m_tree.get_edges_out(id2);
        expr_tree::node_id_t id3 = e2[0];

        m_tree.graph::add(id0, id3);
        m_tree.erase(id1);
        m_tree.erase(id2);
    }
};

const char node_renderer::k_clazz[] = "node_renderer";

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

    const expr_tree::edge_list_t &clist = m_tree.get_edges_out(head);
    if (clist.size() != 2) {
        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Invalid assignment.");
    }

    const node &lhs = m_tree.get_vertex(clist[0]);
    const node &rhs = m_tree.get_vertex(clist[1]);
    if (lhs.get_n() != rhs.get_n()) {
        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Invalid dimensions of lhs and rhs.");
    }

    if (lhs.get_op().compare(node_ident_base::k_op_type)) {
        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Invalid lhs.");
    }
    const node_ident_base &n = lhs.recast_as<node_ident_base>();
    if (n.get_t() != typeid(double)) {
        throw not_implemented("iface", k_clazz, method, __FILE__, __LINE__);
    }

    std::cout << "render expression" << std::endl;
    print_tree(m_tree, std::cout);
    node_renderer(m_tree, head).render();

    std::cout << "rendered expression" << std::endl;
    print_tree(m_tree, std::cout);
}


} // namespace iface
} // namespace libtensor
