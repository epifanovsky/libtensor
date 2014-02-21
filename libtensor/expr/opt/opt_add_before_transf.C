#include <vector>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_transform.h>
#include "opt_add_before_transf.h"

namespace libtensor {
namespace expr {


void opt_add_before_transf(graph &g) {

    typedef graph::node_id_t node_id_t;

    while(true) {

        std::vector<node_id_t> match;

        //  Find all subexpressions like this:
        //  ( Tr ( + E1 E2 ... ) )

        for(graph::iterator i = g.begin(); i != g.end(); ++i) {
            if(!g.get_vertex(i).check_type<node_transform_base>()) continue;
            const graph::edge_list_t &eo = g.get_edges_out(i);
            if(eo.size() != 1) continue;
            if(g.get_vertex(eo[0]).check_type<node_add>()) {
                match.push_back(g.get_id(i));
            }
        }

        if(match.empty()) break;

        //  Transform ( Tr ( + ... ) ) to ( + (Tr .) (Tr .) ... )

        for(size_t i = 0; i < match.size(); i++) {

            node_id_t nidi = match[i]; // Transform node
            node_id_t nidj = g.get_edges_out(nidi).at(0); //!< Add node

            g.erase(nidi, nidj);
            const graph::edge_list_t &ei = g.get_edges_in(nidi);
            for(size_t j = 0; j < ei.size(); j++) g.replace(ei[j], nidi, nidj);

            const node_transform_base &n =
                g.get_vertex(nidi).recast_as<node_transform_base>();
            const graph::edge_list_t &eo = g.get_edges_out(nidj);
            for(size_t j = 0; j < eo.size(); j++) {
                node_id_t nidk = g.add(n), nidx = eo[j];
                g.replace(nidj, eo[j], nidk);
                g.add(nidk, nidx);
            }
        }

        for(size_t i = 0; i < match.size(); i++) g.erase(match[i]);
    }
}


} // namespace expr
} // namespace libtensor
