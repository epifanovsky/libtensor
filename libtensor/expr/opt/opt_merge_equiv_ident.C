#include <vector>
#include <libtensor/expr/dag/node_ident.h>
#include "opt_merge_equiv_ident.h"

namespace libtensor {
namespace expr {


void opt_merge_equiv_ident(graph &g) {

    typedef graph::node_id_t node_id_t;

    std::vector<node_id_t> map_from, map_to;

    //  Make a list of all identity nodes

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {
        if(g.get_vertex(i).get_op().compare(node_ident::k_op_type) == 0) {
            map_from.push_back(g.get_id(i));
        }
    }
    map_to.reserve(map_from.size());

    //  For each identity node try to find equivalent nodes

    for(size_t i = 0; i < map_from.size(); i++) {
        const node_ident &ni =
            g.get_vertex(map_from[i]).recast_as<node_ident>();
        map_to.push_back(map_from[i]);
        for(size_t j = 0; j < i; j++) {
            const node_ident &nj =
                g.get_vertex(map_from[j]).recast_as<node_ident>();
            if(ni.equals(nj)) {
                map_to[i] = map_from[j]; break;
            }
        }
    }

    //  Remap equivalent nodes

    for(size_t i = 0; i < map_from.size(); i++) {
        if(map_from[i] != map_to[i]) {
            const graph::edge_list_t &e = g.get_edges_in(map_from[i]);
            for (size_t k = 0; k < e.size(); k++) {
                g.replace(e[k], map_from[i], map_to[i]);
            }
        }
    }

    //  Remove unlinked nodes

    for(size_t i = 0; i < map_from.size(); i++) {
        if(map_from[i] != map_to[i]) g.erase(map_from[i]);
    }
}


} // namespace expr
} // namespace libtensor
