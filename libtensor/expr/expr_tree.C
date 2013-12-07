#include <set>
#include <libtensor/not_implemented.h>
#include "expr_tree.h"

namespace libtensor {
namespace expr {


expr_tree::node_id_t expr_tree::add(node_id_t id, const node &n) {

    node_id_t idn = graph::add(n);
    graph::add(id, idn);

    return idn;
}


expr_tree::node_id_t expr_tree::add(node_id_t id, const expr_tree &subtree) {

    std::map<node_id_t, node_id_t> map;
    for (iterator i = subtree.begin(); i != subtree.end(); i++) {

        map[subtree.get_id(i)] = graph::add(subtree.get_vertex(i));
    }

    for (iterator i = subtree.begin(); i != subtree.end(); i++) {

        node_id_t cid = map[subtree.get_id(i)];

        const edge_list_t &out = subtree.get_edges_out(i);
        for (size_t j = 0; j < out.size(); j++) {
            graph::add(cid, map[out[j]]);
        }
    }

    node_id_t idn = map[subtree.get_root()];
    graph::add(id, idn);
    return idn;
}


expr_tree::node_id_t expr_tree::insert(node_id_t id, const node &n) {

    node_id_t idn = graph::add(n);

    edge_list_t in(graph::get_edges_in(id));
    for (size_t j = 0; j < in.size(); j++) {

        graph::replace(in[j], id, idn);
    }

    graph::add(idn, id);
    if (id == m_root) m_root = idn;

    return idn;
}


void expr_tree::erase_subtree(node_id_t h) {

    edge_list_t out(graph::get_edges_out(h));
    for (size_t i = 0; i < out.size(); i++) {
        if (graph::get_edges_in(out[i]).size() == 1) {
            erase_subtree(out[i]);
        }
        else {
            graph::erase(h, out[i]);
        }
    }
    if (h != m_root) {
        graph::erase(h);
    }
}


bool expr_tree::move(node_id_t h1, node_id_t h2) {

    if (is_connected(h1, h2)) return false;

    edge_list_t e(get_edges_in(h1));
    for (size_t i = 0; i < e.size(); i++) erase(e[i], h1);
    graph::add(h2, h1);

    return true;
}


bool expr_tree::replace(node_id_t h1, node_id_t h2) {

    if (is_connected(h2, h1)) return false;

    const edge_list_t &e = get_edges_in(h1);
    for (size_t i = 0; i < e.size(); i++) graph::add(e[i], h2);
    erase_subtree(h1);

    return true;
}


expr_tree expr_tree::get_subtree(node_id_t h) const {

    expr_tree tr(get_vertex(h));
    std::map<node_id_t, node_id_t> map;
    std::set<node_id_t> todo, touched;
    map.insert(std::pair<node_id_t, node_id_t>(h, tr.get_root()));
    touched.insert(h);
    todo.insert(h);

    while (! todo.empty()) {
        std::set<node_id_t>::iterator i = todo.begin();

        const edge_list_t &e = get_edges_out(*i);
        for (size_t j = 0; j < e.size(); j++) {
            if (touched.count(e[j]) == 0) {
                node_id_t ei = tr.add(map[*i], get_vertex(e[j]));
                map.insert(std::pair<node_id_t, node_id_t>(e[j], ei));
                touched.insert(e[j]);
                todo.insert(e[j]);
            }
            else {
                tr.graph::add(map[*i], map[e[j]]);
            }
        }
        todo.erase(i);
    }

    return tr;
}

} // namespace expr
} // namespace libtensor
