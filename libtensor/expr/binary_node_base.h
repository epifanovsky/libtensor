#ifndef LIBTENSOR_EXPR_BINARY_NODE_BASE_H
#define LIBTENSOR_EXPR_BINARY_NODE_BASE_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Base class of binary tensor operation node of the expression tree

    \ingroup libtensor_expr
 **/
class binary_node_base : public node {
private:
    const node &m_left; //!< Left argument
    const node &m_right; //!< Right argument

public:
    /** \brief Creates an identity node
        \param op Operation name
        \param left Left argument
        \param right Right argument
     **/
    binary_node_base(const std::string &op, const node &left,
        const node &right) : node(op), m_left(left), m_right(right)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~binary_node_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the left argument of multiplication
     **/
    const node &get_left_arg() const {
        return m_left;
    }

    /** \brief Returns the right argument of multiplication
     **/
    const node &get_right_arg() const {
        return m_right;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_BINARY_NODE_BASE_H
