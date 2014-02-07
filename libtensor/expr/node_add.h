#ifndef LIBTENSOR_EXPR_NODE_ADD_H
#define LIBTENSOR_EXPR_NODE_ADD_H

#include <map>
#include <libtensor/expr/dag/node.h>

namespace libtensor {
namespace expr {


/** \brief Tensor addition node of the expression tree

    Node for adding expression subtrees

    \ingroup libtensor_expr
 **/
class node_add : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates an addition node
        \param n Order of result.
     **/
    node_add(size_t n) : node(node_add::k_op_type, n)
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
