#ifndef LIBTENSOR_EXPR_NODE_DIRSUM_H
#define LIBTENSOR_EXPR_NODE_DIRSUM_H

#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Direct tensor sum node of the expression tree

    \ingroup libtensor_expr
 **/
class node_dirsum : public nary_node_base {
public:
    /** \brief Creates an direct tensor sum node of two tensors
        \param arg1 First argument
        \param arg2 Second argument
     **/
    node_dirsum(const node &arg1, const node &arg2) :
        nary_node_base("dirsum", arg1, arg2)
    { }

    /** \brief Creates an direct tensor sum node of n tensors
        \param args List of arguments
     **/
    node_dirsum(std::vector<const node *> &args) :
        nary_node_base("dirsum", args)
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
