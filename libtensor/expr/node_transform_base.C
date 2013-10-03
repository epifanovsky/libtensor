#include "node_transform_base.h"

namespace libtensor {
namespace expr {


void node_transform_base::check() {

#ifdef LIBTENSOR_DEBUG
    std::vector<bool> ok(m_perm.size(), false);
    for(size_t i = 0; i < m_perm.size(); i++) {
        if (ok[m_perm[i]]) {
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index duplicate.");
        }

        ok[m_perm[i]] = true;
    }

    for(size_t i = 0; i < m_perm.size(); i++) {
        if(!ok[m_perm[i]])
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index missing.");
    }
#endif // LIBTENSOR_DEBUG
}


} // namespace expr
} // namespace libtensor
