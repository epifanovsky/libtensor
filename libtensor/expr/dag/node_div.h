#ifndef LIBTENSOR_EXPR_NODE_DIV_H
#define LIBTENSOR_EXPR_NODE_DIV_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: element-wise division

    \ingroup libtensor_expr_dag
 **/
class node_div : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates the node
        \param n Tensor order.
     **/
    node_div(size_t n) :
        node(k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_div() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_div(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIV_H
