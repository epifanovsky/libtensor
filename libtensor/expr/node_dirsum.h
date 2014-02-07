#ifndef LIBTENSOR_EXPR_NODE_DIRSUM_H
#define LIBTENSOR_EXPR_NODE_DIRSUM_H

#include <libtensor/expr/dag/node.h>

namespace libtensor {
namespace expr {


/** \brief Direct tensor sum node of the expression tree

    \ingroup libtensor_expr
 **/
class node_dirsum : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates an direct tensor sum node of two tensors
        \param arg1 First argument
        \param arg2 Second argument
     **/
    node_dirsum(size_t n) : node(k_op_type, n) { }

    /** \brief Virtual destructor
     **/
    virtual ~node_dirsum() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_dirsum *clone() const {
        return new node_dirsum(*this);
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIRSUM_H
