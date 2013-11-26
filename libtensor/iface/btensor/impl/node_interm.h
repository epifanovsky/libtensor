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
    struct counter {
        btensor_placeholder<N, T> bt; //!< Intermediate
        size_t cnt; //!< Counter

        counter() : cnt(1) { }
    };
    counter *m_cnt;

public:
    node_interm() : node_interm_base(N) {
        m_cnt = new counter();
    }

    node_interm(const node_interm<N, T> &other) : node_interm_base(N) {
        m_cnt = other.m_cnt;
        m_cnt->cnt++;
    }

    virtual ~node_interm() {
        m_cnt->cnt--;
        if (m_cnt->cnt == 0) delete m_cnt;
    }

    virtual node_interm *clone() const {
        return new node_interm(*this);
    }

    virtual const std::type_info &get_t() const {
        return typeid(T);
    }

    iface::any_tensor<N, T> &get_tensor() const {
        return m_cnt->bt;
    }

private:
    node_interm<N, T> &operator==(const node_interm<N, T> &);
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_NODE_INTERM_H
