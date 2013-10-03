#ifndef LIBTENSOR_EXPR_NODE_DIRSUM_H
#define LIBTENSOR_EXPR_NODE_DIRSUM_H

#include "binary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Direct tensor sum node of the expression tree

    \ingroup libtensor_expr
 **/
class node_dirsum : public binary_node_base {
public:
    /** \brief Creates an identity node
        \param tid Tensor ID.
     **/
    node_dirsum(const node &left, const node &right) :
        node("dirsum", left, right)
    { }

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
