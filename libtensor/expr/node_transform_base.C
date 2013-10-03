#include "node_transform_base.h"

namespace libtensor {
namespace expr {


void node_transform_base::check() throw(exception) {

#ifdef LIBTENSOR_DEBUG
    std::vector<bool> ok(m_order.size(), false);
    for (size_t i = 0; i < m_order.size(); i++) {
        if (ok[m_order[i]]) {
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index duplicate.");
        }

        ok[m_order[i]] = true;
    }

    for (size_t i = 0; i < m_order.size(); i++) {
        if (! ok[m_order[i]])
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index missing.");
    }
#endif
}


} // namespace expr
} // namespace libtensor
