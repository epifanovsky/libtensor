#ifndef LIBTENSOR_EXPR_NODE_ADD_H
#define LIBTENSOR_EXPR_NODE_ADD_H

#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor addition node of the expression tree

    Node for adding two expression subtrees

    \ingroup libtensor_expr
 **/
class node_add : public nary_node_base {
public:
    /** \brief Creates an addition node
        \param arg1 First argument.
        \param arg2 Second argument.
     **/
    node_add(const node &arg1, const node &arg2) :
        nary_node_base("add", arg1.get_n(), arg1, arg2)
    { }

    /** \brief Creates an addition node
        \param args List of arguments.
     **/
    node_add(const std::vector<const node *> &args) :
        nary_node_base("add", args.empty() ? 0 : args[0]->get_n(), args)
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
