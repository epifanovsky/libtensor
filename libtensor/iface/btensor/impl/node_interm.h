#ifndef LIBTENSOR_IFACE_NODE_INTERM_H
#define LIBTENSOR_IFACE_NODE_INTERM_H

#include <map>
#include <libtensor/expr/node.h>
#include "btensor_placeholder.h"

namespace libtensor {
namespace iface {


class node_interm_base : public expr::node {
public:
    static const char k_op_type[]; //!< Operation type

    node_interm_base(size_t n) : expr::node(k_op_type, n) { }

    virtual ~node_interm_base() { }

    virtual const std::type_info &get_t() const = 0;
};


template<size_t N, typename T>
class node_interm : public node_interm_base {
private:
    btensor_placeholder<N, T> *m_bt; //!< Intermediate
    size_t m_cnt; //!< Counter

public:
    node_interm();

    virtual ~node_interm() { }

    virtual node_interm *clone() const {
        return new node_interm(*this);
    }

    virtual const std::type_info &get_t() const {
        return typeid(T);
    }

    btensor_placeholder<N, T> &get_placeholder() {
        return m_bt;
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_NODE_INTERM_H
