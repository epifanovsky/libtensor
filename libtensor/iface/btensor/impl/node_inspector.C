#include "node_inspector.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


const node_ident &node_inspector::extract_ident() const {

    if(m_node.get_op().compare("ident") == 0) {

        return m_node.recast_as<node_ident>();

    } else if(m_node.get_op().compare("transform") == 0) {

        const node_transform_base &nb = m_node.recast_as<node_transform_base>();
        return node_inspector(nb.get_arg()).extract_ident();

    }

    throw "No identity node";
}


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
