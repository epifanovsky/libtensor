#ifndef LIBTENSOR_EXPR_NODE_DIRPROD_H
#define LIBTENSOR_EXPR_NODE_DIRPROD_H

#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Direct tensor product node of the expression tree

    \ingroup libtensor_expr
 **/
class node_dirprod : public nary_node_base {
public:
    /** \brief Creates an direct tensor product node of two tensors
        \param arg1 First argument
        \param arg2 Second argument
     **/
    node_dirprod(const node &arg1, const node &arg2) :
        nary_node_base("dirprod", arg1, arg2)
    { }

    /** \brief Creates an direct tensor product node of n tensors
        \param args List of arguments
     **/
    node_dirprod(std::vector<const node *> &args) :
        nary_node_base("dirprod", args)
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
