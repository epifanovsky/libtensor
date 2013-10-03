#ifndef LIBTENSOR_EXPR_NODE_DIRPROD_H
#define LIBTENSOR_EXPR_NODE_DIRPROD_H

#include "binary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Direct tensor product node of the expression tree

    \ingroup libtensor_expr
 **/
class node_dirprod : public binary_node_base {
public:
    /** \brief Creates an identity node
        \param tid Tensor ID.
     **/
    node_dirprod(const node &left, const node &right) :
        node("dirprod", left, right)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_dirprod() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_dirprod *clone() const {
        return new node_dirprod(*this);
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIRPROD_H
