#include <deque>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_ident.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/expr/opt/opt_add_before_transf.h>
#include <libtensor/expr/opt/opt_merge_adjacent_add.h>
#include <libtensor/expr/opt/opt_merge_adjacent_transf.h>
#include <libtensor/expr/opt/opt_merge_equiv_ident.h>
#include "node_interm.h"
#include "eval_tree_builder_btensor.h"

namespace libtensor {
namespace expr {
using namespace eval_btensor_double; // for dispatch_1


const char eval_tree_builder_btensor::k_clazz[] = "eval_tree_builder_btensor";


namespace {

typedef graph::node_id_t node_id_t;

/** \brief Inserts an intermediate assignment at current position in graph
 **/
class interm_inserter {
public:
    enum {
        Nmax = eval_tree_builder_btensor::Nmax
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
        node_id_t id1 = m_g.add(node_interm<N, double>());

        graph::edge_list_t ei = m_g.get_edges_in(m_nid);
        for(size_t i = 0; i < ei.size(); i++) m_g.replace(ei[i], m_nid, id0);

        m_g.add(id0, id1);
        m_g.add(id0, m_nid);
    }

};

void assume_adds(graph &g) {

    std::vector<node_id_t> replace, erase;

    //  Find all nodes to replace
    //  ( assign X ( + E1 E2 ... ) )
    //  with
    //  ( assign X E1 E2 ... )

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {
        if(!g.get_vertex(i).check_type<node_assign>()) continue;
        const graph::edge_list_t &eo = g.get_edges_out(i);
        if(eo.size() <= 1) continue;
        bool all_adds = true;
        for(size_t j = 1; j < eo.size(); j++) {
            if(!g.get_vertex(eo[j]).check_type<node_add>()) all_adds = false;
        }
        if(all_adds) replace.push_back(g.get_id(i));
    }

    //  Perform replacement

    for(size_t i = 0; i < replace.size(); i++) {
        graph::edge_list_t eo1 = g.get_edges_out(replace[i]);
        for(size_t j = 1; j < eo1.size(); j++) {
            g.erase(replace[i], eo1[j]);
            erase.push_back(eo1[j]);
        }
        for(size_t j = 1; j < eo1.size(); j++) {
            graph::edge_list_t eo2 = g.get_edges_out(eo1[j]);
            for(size_t k = 0; k < eo2.size(); k++) {
                g.erase(eo1[j], eo2[k]);
                g.add(replace[i], eo2[k]);
            }
        }
    }

    for(size_t i = 0; i < erase.size(); i++) g.erase(erase[i]);
}

void insert_intermediates(graph &g, graph::node_id_t n0) {

    if(!g.get_vertex(n0).check_type<node_assign>()) {
        throw eval_exception(__FILE__, __LINE__, "eval",
            "eval_tree_builder_btensor", "insert_intermediates()",
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

    if(g.get_vertex(n).check_type<node_assign>()) order.push_back(n);
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
