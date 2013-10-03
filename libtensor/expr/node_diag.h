#ifndef LIBTENSOR_EXPR_NODE_DIAG_H
#define LIBTENSOR_EXPR_NODE_DIAG_H

#include <vector>
#include "unary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor diagonal node of the expression tree

    The tensor diagonal is taken over the indexes specified by diagdims.

    \ingroup libtensor_expr
 **/
class node_diag : public unary_node_base {
private:
    std::vector<size_t> m_ddims; //!< Dimensions to take the diagonal from

public:
    /** \brief Creates an identity node
        \param tid Tensor ID.
     **/
    node_diag(const node &arg, const std::vector<size_t> &diagdims) :
        node("diag", arg), m_ddims(diagdims)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_diag() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_diag *clone() const {
        return new node_diag(*this);
    }

    /** \brief Returns the dimensions the diagonal is taken off
     **/
    const std::vector<size_t> &get_diag_dims() const {
        return m_ddims;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIAG_H
