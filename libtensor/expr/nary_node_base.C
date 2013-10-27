#include <libtensor/exception.h>
#include "nary_node_base.h"

namespace libtensor {
namespace expr {


nary_node_base::nary_node_base(const std::string &op,
    const std::vector<const node *> &args) : node(op), m_args(args.size()) {

    std::vector<const node *>::iterator ito = m_args.begin();
    for (std::vector<const node *>::const_iterator ifro = args.begin();
            ifro != args.end(); ifro++) {

        *ito = (*ifro)->clone();
    }
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
