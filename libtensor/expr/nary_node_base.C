#include <libtensor/exception.h>
#include "nary_node_base.h"

namespace libtensor {
namespace expr {


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
