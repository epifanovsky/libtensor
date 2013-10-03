#ifndef LIBTENSOR_EXPR_NODE_ADD_H
#define LIBTENSOR_EXPR_NODE_ADD_H

#include "binary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor addition node of the expression tree

    Node for adding two expression subtrees

    \ingroup libtensor_expr
 **/
class node_add : public binary_node_base {
public:
    /** \brief Creates an identity node
        \param tid Tensor ID.
     **/
    node_add(const node &left, const node &right) :
        binary_node_base("add", left, right)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_add() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_add *clone() const {
        return new node_add(*this);
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_ADD_H
