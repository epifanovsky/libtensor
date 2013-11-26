#ifndef LIBTENSOR_IFACE_NODE_INTERM_H
#define LIBTENSOR_IFACE_NODE_INTERM_H

#include <map>
#include <libtensor/expr/node_ident.h>
#include "btensor_placeholder.h"

namespace libtensor {
namespace iface {


template<size_t N, typename T>
class node_interm : public expr::node_ident<N, T> {
private:
    btensor_placeholder<N, T> m_bt; //!< Intermediate

public:
    node_interm() : expr::node_ident<N, T>(m_bt) { }

    virtual ~node_interm() { }

    btensor_placeholder<N, T> &get_placeholder() {
        return m_bt;
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_NODE_INTERM_H
