#ifndef LIBTENSOR_EXPR_NODE_ASSIGN_H
#define LIBTENSOR_EXPR_NODE_ASSIGN_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: assignment

    Assignment is a binary operation that represents the evaluation of the
    right-hand-side (rhs) and placement of the result into the left-hand-side
    (lhs). The first argument is lhs, the second argument is rhs.
    The expression on the lhs must be assignable (e.g. node_ident).

    \sa node, node_ident

    \ingroup libtensor_expr_dag
 **/
class node_assign : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    bool m_add; //!< Assign with addition (true/false)

public:
    /** \brief Creates an assignment node
        \param n Tensor order
        \param add Assign with addition (+=)
     **/
    node_assign(size_t n, bool add) :
        node(k_op_type, n), m_add(add)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_assign() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_assign(*this);
    }

    bool is_add() const {
        return m_add;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_ASSIGN_H
