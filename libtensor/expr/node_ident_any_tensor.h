#ifndef LIBTENSOR_EXPR_NODE_IDENT_ANY_TENSOR_H
#define LIBTENSOR_EXPR_NODE_IDENT_ANY_TENSOR_H

#include <libtensor/expr/dag/node_ident.h>
#include <libtensor/iface/any_tensor.h>

namespace libtensor {
namespace expr {


/** \brief Tensor identity node of the expression tree

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class node_ident_any_tensor : public node_ident {
private:
    iface::any_tensor<N, T> &m_t;

public:
    /** \brief Creates an identity node
     **/
    node_ident_any_tensor(iface::any_tensor<N, T> &t) : node_ident(N), m_t(t)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ident_any_tensor() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_ident_any_tensor *clone() const {
        return new node_ident_any_tensor(*this);
    }

    /** \brief Returns the element type in the tensor
     **/
    virtual const std::type_info &get_type() const {
        return typeid(T);
    }

    /** \brief Returns the tensor
     **/
    iface::any_tensor<N, T> &get_tensor() const {
        return m_t;
    }

    /** \brief Checks if both identity nodes contain the same tensor
     **/
    virtual bool equals(const node_ident &n) const;

private:
    bool tensor_equals(any_tensor<N, T> &t) {
        return m_t == t;
    }
};


template<size_t N, typename T>
bool node_ident_any_tensor<N, T>::equals(const node_ident &n) const {

    if (n.get_n() == N || n.get_type() == typeid(T)) {
        iface::any_tensor<N, T> &t2 =
                static_cast<const node_ident_any_tensor<N, T> &>(n).get_tensor();
        return &m_t == &t2;
    }

    return false;
}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_IDENT_ANY_TENSOR_H
