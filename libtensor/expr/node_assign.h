#ifndef LIBTENSOR_EXPR_NODE_ASSIGN_H
#define LIBTENSOR_EXPR_NODE_ASSIGN_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Assignment node of the expression tree

    \ingroup libtensor_expr
 **/
class node_assign : public node {
private:
    tid_t m_tid; //!< Tensor ID
    node *m_rhs; //!< Operation to be assigned
    bool m_add; //!< Assignment under addition (A += B)

public:
    /** \brief Creates an assignment node
     **/
    node_assign(tid_t tid, const node &rhs, bool add = false) :
        node("assign", rhs.get_n()), m_tid(tid), m_rhs(rhs.clone()),
        m_add(add)
    { }

    /** \brief Copy constructor
     **/
    node_assign(const node_assign &n) :
        node(n), m_tid(n.m_tid), m_rhs(n.m_rhs->clone()), m_add(n.m_add)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_assign() {
        delete m_rhs;
    }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_assign *clone() const {
        return new node_assign(*this);
    }

    /** \brief Returns the tensor ID on the left-hand side of the assignment
     **/
    tid_t get_tid() const {
        return m_tid;
    }

    /** \brief Returns the right-hand side of the assignment
     **/
    const node &get_rhs() const {
        return *m_rhs;
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
