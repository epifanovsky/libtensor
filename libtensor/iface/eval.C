#include <libtensor/expr/node_ident_base.h>
#include "eval.h"

namespace libtensor {
namespace iface {
using expr::expr_tree;
using expr::graph;
using expr::node;


bool eval::can_evaluate(const expr_tree &e) const {

    const eval_i *ev = find_evaluator(e, e.get_root(), e);
    return (ev != 0);
}


void eval::evaluate(const expr_tree &e) const {

    const eval_i *ev = find_evaluator(e, e.get_root(), e);
    if(ev == 0) {
        throw 0;
    }
    ev->evaluate(e);
}


const eval_i *eval::find_evaluator(const graph &g, graph::node_id_t nid,
    const expr_tree &e) const {

    /*
    const node &n = g.get_vertex(nid);

    if(n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident_base &n1 = n.recast_as<node_ident_base>();
    }

    const expr_tree::edge_list_t &out = g.get_edges_out(nid);
    for(expr_tree::edge_list_t::const_iterator i = out.begin(); i != out.end();
        ++i) {

        graph::node_id_t nid1 = g.get_id(i);
    }
    */
    return 0;
}


} // namespace iface
} // namespace libtensor
