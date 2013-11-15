#ifndef LIBTENSOR_EXPR_NODE_DIV_H
#define LIBTENSOR_EXPR_NODE_DIV_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Element-wise tensor division node of expression tree

    \ingroup libtensor_expr
 **/
class node_div : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates an identity node
        \param arg1 Left argument.
        \param arg2 Right argument.
     **/
    node_div(size_t n) :
        node(node_div::k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_div() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_div *clone() const {
        return new node_div(*this);
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_MULT_H
