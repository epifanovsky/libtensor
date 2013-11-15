#ifndef LIBTENSOR_EXPR_NODE_ASSIGN_H
#define LIBTENSOR_EXPR_NODE_ASSIGN_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Assignment node of the expression tree

    \ingroup libtensor_expr
 **/
class node_assign : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    bool m_add; //!< Assignment under addition (A += B)

public:
    /** \brief Creates an assignment node
        \param n Tensor order
        \param add Add to result
     **/
    node_assign(size_t n, bool add = false) :
        node(node_assign::k_op_type, n), m_add(add) { }

    /** \brief Virtual destructor
     **/
    virtual ~node_assign() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_assign *clone() const {
        return new node_assign(*this);
    }

    /** \brief Returns whether this is an add-to assignment
     **/
    bool is_add() const {
        return m_add;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_ASSIGN_H
