#include <deque>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_const_scalar.h>
#include <libtensor/expr/dag/node_ident.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_scale.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/expr/opt/opt_add_before_transf.h>
#include <libtensor/expr/opt/opt_merge_adjacent_add.h>
#include <libtensor/expr/opt/opt_merge_adjacent_transf.h>
#include <libtensor/expr/opt/opt_merge_equiv_ident.h>
#include "node_ctf_btensor_interm.h"
#include "eval_tree_builder_ctf_btensor.h"

namespace libtensor {
namespace expr {
using eval_btensor_double::dispatch_1;


const char eval_tree_builder_ctf_btensor::k_clazz[] =
    "eval_tree_builder_ctf_btensor";


namespace {

typedef graph::node_id_t node_id_t;

/** \brief Inserts an intermediate assignment at current position in graph
 **/
class interm_inserter {
public:
    enum {
        Nmax = eval_tree_builder_ctf_btensor::Nmax
    };

private:
    graph &m_g; //!< Expression DAG
    node_id_t m_nid; //!< ID of node

public:
    interm_inserter(graph &g, node_id_t nid) :
        m_g(g), m_nid(nid)
    { }

    void add() {
        dispatch_1<1, Nmax>::dispatch(*this, m_g.get_vertex(m_nid).get_n());
    }

    template<size_t N>
    void dispatch() {
        add<N>();
    }

private:
    template<size_t N>
    void add() {

        node_id_t id0 = m_g.add(node_assign(m_g.get_vertex(m_nid).get_n(),
                false));
        node_id_t id1 = m_g.add(node_ctf_btensor_interm<N, double>());

        graph::edge_list_t ei = m_g.get_edges_in(m_nid);
        for(size_t i = 0; i < ei.size(); i++) m_g.replace(ei[i], m_nid, id0);

        m_g.add(id0, id1);
        m_g.add(id0, m_nid);
    }

};

void insert_intermediates(graph &g, graph::node_id_t n0) {

    if(g.get_vertex(n0).check_type<node_scale>()) return;

    if(!g.get_vertex(n0).check_type<node_assign>()) {
        throw eval_exception(__FILE__, __LINE__, "eval",
            "eval_tree_builder_ctf_btensor", "insert_intermediates()",
            "Expected an assignment node");
    }

    typedef graph::node_id_t node_id_t;

    std::deque< std::pair<node_id_t, int> > q;

    const graph::edge_list_t &eo = g.get_edges_out(n0);
    for(size_t i = 1; i < eo.size(); i++) q.push_back(std::make_pair(eo[i], 1));

    while(!q.empty()) {

        //  First element in the pair is currently processed node,
        //  second element keeps track of subtree level to enable skipping
        //  the formation of intermediates for add, symm, etc.

        std::pair<node_id_t, int> p = q.front();
        q.pop_front();
        node_id_t n = p.first;
        int l = p.second;

        //  Skip nodes that won't need further inspection
        if(g.get_vertex(n).check_type<node_ident>() ||
            g.get_vertex(n).check_type<node_const_scalar_base>() ||
            g.get_vertex(n).check_type<node_scalar_base>() ||
            g.get_vertex(n).check_type<node_assign>()) continue;

        //  Inspect children nodes further
        int l1 = 0;
        if(g.get_vertex(n).check_type<node_transform_base>()) {
            l1 = l;
        } else if(g.get_vertex(n).check_type<node_add>()) {
            l1 = 1;
        } else if(g.get_vertex(n).check_type<node_symm_base>()) {
            l1 = 1;
        } else {
            if(l > 0) l1 = l - 1;
        }

        const graph::edge_list_t &eo = g.get_edges_out(n);
        for(size_t i = 0; i < eo.size(); i++) {
            q.push_back(std::make_pair(eo[i], l1));
        }

        //  Skip transformation nodes
        if(g.get_vertex(n).check_type<node_transform_base>()) continue;

        //  Otherwise insert an intermediate
        if(l == 0) interm_inserter(g, n).add();
    }
}

void make_eval_order_depth_first(graph &g, node_id_t n,
    std::vector<node_id_t> &order) {

    const graph::edge_list_t &eo = g.get_edges_out(n);
    for(size_t i = 0; i < eo.size(); i++) {
        make_eval_order_depth_first(g, eo[i], order);
    }

    if(g.get_vertex(n).check_type<node_assign>() ||
        g.get_vertex(n).check_type<node_scale>()) {

        order.push_back(n);
    }
}

} // unnamed namespace


void eval_tree_builder_ctf_btensor::build() {

    static const char method[] = "build()";

    //  For now assume it's double
    //  TODO: implement other types

    expr_tree::node_id_t head = m_tree.get_root();
    const node &hnode = m_tree.get_vertex(head);

    if(hnode.get_op() != node_assign::k_op_type &&
        hnode.get_op() != node_scale::k_op_type) {

        throw bad_parameter("iface", k_clazz, method,
                __FILE__, __LINE__, "Unexpected root node.");
    }

    opt_merge_equiv_ident(m_tree);
    opt_merge_adjacent_transf(m_tree);
    opt_add_before_transf(m_tree);
    opt_merge_adjacent_transf(m_tree);
    opt_merge_adjacent_add(m_tree);

    insert_intermediates(m_tree, m_tree.get_root());

    make_eval_order_depth_first(m_tree, m_tree.get_root(), m_order);
}


} // namespace expr
} // namespace libtensor
