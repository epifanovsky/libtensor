#include <vector>
#include <libtensor/expr/dag/node_add.h>
#include "opt_merge_adjacent_add.h"

namespace libtensor {
namespace expr {


void opt_merge_adjacent_add(graph &g) {

    typedef graph::node_id_t node_id_t;

    std::vector<node_id_t> erase;

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {

        if(!g.get_vertex(i).check_type<node_add>()) continue;

        node_id_t nid = g.get_id(i);
        bool repeat;
        do {
            repeat = false;

            std::vector<node_id_t> out; // New out edges
            graph::edge_list_t eo0 = g.get_edges_out(i);
            for(size_t j = 0; j < eo0.size(); j++) {
                if(g.get_vertex(eo0[j]).check_type<node_add>()) {
                    graph::edge_list_t eo1 = g.get_edges_out(eo0[j]);
                    for(size_t k = 0; k < eo1.size(); k++) {
                        out.push_back(eo1[k]);
                        g.erase(eo0[j], eo1[k]);
                    }
                    repeat = true;
                    g.erase(nid, eo0[j]);
                    erase.push_back(eo0[j]);
                } else {
                    out.push_back(eo0[j]);
                }
            }

            if(repeat) {
                for(size_t j = 0; j < eo0.size(); j++) g.erase(nid, eo0[j]);
                for(size_t j = 0; j < out.size(); j++) g.add(nid, out[j]);
            }

        } while(repeat);
    }

    for(size_t i = 0; i < erase.size(); i++) g.erase(erase[i]);
}


} // namespace expr
} // namespace libtensor
