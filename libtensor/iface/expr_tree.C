#include <libtensor/not_implemented.h>
#include "expr_tree.h"

namespace libtensor {
namespace iface {


void expr_tree::add(node_id_t id, const expr::node &n) {

    node_id_t idn = expr::graph::add(n);
    expr::graph::add(id, idn);
}


void expr_tree::add(node_id_t id, const expr_tree &subtree) {

    std::map<node_id_t, node_id_t> map;
    for (iterator i = subtree.begin(); i != subtree.end(); i++) {

        map[subtree.get_id(i)] = expr::graph::add(subtree.get_vertex(i));
    }

    for (iterator i = subtree.begin(); i != subtree.end(); i++) {

        node_id_t cid = map[subtree.get_id(i)];

        const edge_list_t &out = subtree.get_edges_out(i);
        for (size_t j = 0; j < out.size(); j++) {
            expr::graph::add(cid, map[out[j]]);
        }
    }

    expr::graph::add(id, map[subtree.get_root()]);
}


void expr_tree::insert(node_id_t id, const expr::node &n) {

    node_id_t idn = expr::graph::add(n);
    expr::graph::add(idn, id);
    if (id == m_root) m_root = idn;

    edge_list_t in(expr::graph::get_edges_in(id));
    for (size_t j = 0; j < in.size(); j++) {
        expr::graph::erase(in[j], id);
        expr::graph::add(in[j], idn);
    }
}


void expr_tree::erase_subtree(node_id_t h) {

    edge_list_t out(expr::graph::get_edges_out(h));
    for (size_t i = 0; i < out.size(); i++) {
        erase_subtree(out[i]);
    }
    if (h != m_root)
        expr::graph::erase(h);

}


bool expr_tree::move(node_id_t h1, node_id_t h2) {

    throw not_implemented("libtensor::iface", "expr_tree",
            "move(node_id_t, node_id_t)", __FILE__, __LINE__);
}


bool expr_tree::replace(node_id_t h1, node_id_t h2) {

    throw not_implemented("libtensor::iface", "expr_tree",
            "replace(node_id_t, node_id_t)", __FILE__, __LINE__);
}


} // namespace iface
} // namespace libtensor
