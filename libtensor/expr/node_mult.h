#ifndef LIBTENSOR_EXPR_NODE_MULT_H
#define LIBTENSOR_EXPR_NODE_MULT_H

#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Element-wise tensor multiplication node of expression tree

    \ingroup libtensor_expr
 **/
class node_mult : public nary_node_base {
private:
    bool m_recip; //!< Perform division

public:
    /** \brief Creates an identity node
        \param left Left argument.
        \param right Right argument.
        \param recip Perform division (left / right).
     **/
    node_mult(const node &left, const node &right, bool recip) :
        nary_node_base("mult", 0, left, right), m_recip(recip)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_mult() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_mult *clone() const {
        return new node_mult(*this);
    }

    /** \brief Returns the right argument of multiplication
     **/
    bool do_recip() const {
        return m_recip;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_MULT_H
