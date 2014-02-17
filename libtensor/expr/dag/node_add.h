#ifndef LIBTENSOR_EXPR_NODE_ADD_H
#define LIBTENSOR_EXPR_NODE_ADD_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: addition

    \ingroup libtensor_expr_dag
 **/
class node_add : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates an addition node
        \param n Order of result.
     **/
    node_add(size_t n) :
        node(k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_add() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_add(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_ADD_H
