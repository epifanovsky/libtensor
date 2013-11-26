#include "print_node.h"
#include "print_tree.h"

namespace libtensor {
namespace expr {


void print_tree(const expr_tree &tr, expr_tree::node_id_t h,
        std::ostream &os, size_t indent) {

    const node &n = tr.get_vertex(h);
    print_node(n, os, indent);

    const expr_tree::edge_list_t &e = tr.get_edges_out(h);
    for (size_t i = 0; i < e.size(); i++) {
        print_tree(tr, e[i], os, indent + 2);
    }
}


} // namespace expr
} // namespace libtensor
