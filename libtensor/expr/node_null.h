#ifndef LIBTENSOR_EXPR_NODE_NULL_H
#define LIBTENSOR_EXPR_NODE_NULL_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: null (empty) expression

    This special node is to be used as a placeholder for creating empty
    expressions.

    \ingroup libtensor_expr
 **/
class node_null : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates a null node
        \param n Tensor order
     **/
    node_null(size_t n) : node(k_op_type, n) { }

    /** \brief Virtual destructor
     **/
    virtual ~node_null() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_null(*this);
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_NULL_H
