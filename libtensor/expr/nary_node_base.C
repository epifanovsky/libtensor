#include <libtensor/exception.h>
#include "nary_node_base.h"

namespace libtensor {
namespace expr {


nary_node_base::nary_node_base(
    const std::string &op, size_t n,
    const std::vector<const node*> &args) :

    node(op, n), m_args(args.size()) {

    for(size_t i = 0; i < args.size(); i++) m_args[i] = args[i]->clone();
}


nary_node_base::nary_node_base(const nary_node_base &n) :
    node(n), m_args(n.m_args.size()) {

    for(size_t i = 0; i < n.m_args.size(); i++) {
        m_args[i] = n.m_args[i]->clone();
    }
}


nary_node_base::~nary_node_base() {

    for(size_t i = 0; i < m_args.size(); i++) delete m_args[i];
}


const node &nary_node_base::get_arg(size_t i) const {

#ifdef LIBTENSOR_DEBUG
    if (i >= m_args.size()) {
        throw bad_parameter(g_ns, "nary_node_base", "get_arg(size_t)",
                __FILE__, __LINE__, "i.");
    }
#endif // LIBTENSOR_DEBUG

    const node *ni = m_args[i];
    return *ni;
}


} // namespace expr
} // namespace libtensor
