#ifndef LIBTENSOR_EXPR_NODE_IDENT_H
#define LIBTENSOR_EXPR_NODE_IDENT_H

#include <libtensor/iface/any_tensor.h>
#include "node_ident_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor identity node of the expression tree

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class node_ident : public node_ident_base {
private:
    iface::any_tensor<N, T> &m_t;

public:
    /** \brief Creates an identity node
     **/
    node_ident(iface::any_tensor<N, T> &t) : node_ident_base(N), m_t(t)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ident() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_ident *clone() const {
        return new node_ident(*this);
    }

    virtual const std::type_info &get_t() const {
        return typeid(T);
    }

    /** \brief Returns the tensor
     **/
    iface::any_tensor<N, T> &get_tensor() const {
        return m_t;
    }

    /** \brief Checks if both identity nodes contain the same tensor
     **/
    virtual bool operator==(const node_ident_base &n) const;

private:
    bool tensor_equals(any_tensor<N, T> &t) {
        return m_t == t;
    }
};


template<size_t N, typename T>
bool node_ident<N, T>::operator==(const node_ident_base &n) const {

    if (n.get_n() == N || n.get_t() == typeid(T)) {
        iface::any_tensor<N, T> &t2 =
                static_cast<const node_ident<N, T> &>(n).get_tensor();
        return &m_t == &t2;
    }

    return false;
}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_IDENT_H
