#ifndef LIBTENSOR_EXPR_NODE_IDENT_H
#define LIBTENSOR_EXPR_NODE_IDENT_H

#include <typeinfo>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: identity

    The identity node contains a reference to the tensor object. This node
    type does not have any arguments. Identity nodes are assignable.

    This class is abstract; derived classes shall specify concrete tensor
    types contained.

    \sa node, node_assign

    \ingroup libtensor_expr_dag
 **/
class node_ident : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates the identity node
        \param n Tensor order
     **/
    node_ident(size_t n) :
        node(k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ident() { }

    /** \brief Returns the element type in the tensor
     **/
    virtual const std::type_info &get_type() const = 0;

    /** \brief Checks if both identity nodes contain the same tensor
     **/
    virtual bool equals(const node_ident &n) const = 0;

    /** \brief Comparison of two identity nodes returns true if they contain
            the same tensor
     **/
    bool operator==(const node_ident &other) const {
        return equals(other);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_IDENT_H
