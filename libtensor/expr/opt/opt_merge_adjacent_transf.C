#include <vector>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_transform.h>
#include "opt_merge_adjacent_transf.h"

namespace libtensor {
namespace expr {


namespace {

typedef graph::node_id_t node_id_t;

void combine_perm(const std::vector<size_t> &p1, const std::vector<size_t> &p2,
    std::vector<size_t> &p) {

    p.clear();
    if(p1.size() != p2.size()) return;

    p.resize(p1.size());
    for(size_t i = 0; i < p2.size(); i++) p[i] = p1[p2[i]];
}

} // unnamed namespace

template<typename T>
void opt_merge_adjacent_transf(graph &g) {

    typedef graph::node_id_t node_id_t;

    std::vector<node_id_t> erase;

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {

        if(!g.get_vertex(i).check_type<node_transform_base>()) continue;
        const graph::edge_list_t &ei = g.get_edges_in(i);
        const graph::edge_list_t &eo = g.get_edges_out(i);
        if(eo.size() != 1) continue;
        if(!g.get_vertex(eo[0]).check_type<node_transform_base>()) continue;

        node_id_t nidi = g.get_id(i), nidj = eo[0];

        const node_transform_base &ni0 =
            g.get_vertex(i).recast_as<node_transform_base>();
        const node_transform_base &nj0 =
            g.get_vertex(eo[0]).recast_as<node_transform_base>();

        if(ni0.get_type() == typeid(T) &&
            nj0.get_type() == typeid(T)) { // do we need this if statement?

            const node_transform<T> &ni =
                ni0.recast_as< node_transform<T> >();
            const node_transform<T> &nj =
                nj0.recast_as< node_transform<T> >();

            std::vector<size_t> perm;
            combine_perm(nj.get_perm(), ni.get_perm(), perm);
            scalar_transf<T> c(nj.get_coeff());
            c.transform(ni.get_coeff());
            g.replace(nidj, node_transform<T>(perm, c));
            g.erase(nidi, nidj);
            for(size_t j = 0; j < ei.size(); j++) g.replace(ei[j], nidi, nidj);
            erase.push_back(nidi);
        }
    }

    for(size_t i = 0; i < erase.size(); i++) g.erase(erase[i]);
}

template void opt_merge_adjacent_transf<double>(graph &);
template void opt_merge_adjacent_transf<float>(graph &);

} // namespace expr
} // namespace libtensor
