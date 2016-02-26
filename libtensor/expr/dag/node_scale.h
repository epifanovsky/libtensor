#ifndef LIBTENSOR_EXPR_NODE_SCALE_H
#define LIBTENSOR_EXPR_NODE_SCALE_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: scaling

    Scaling multiplies the left-hand-side (first argument, node_ident)
    by a constant factor (second argument, node_const_scalar).

    \sa node, node_ident, node_const_scalar

    \ingroup libtensor_expr_dag
 **/
class node_scale : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates a scaling node
        \param n Tensor order
     **/
    node_scale(size_t n) :
        node(k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_scale() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_scale(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SCALE_H
