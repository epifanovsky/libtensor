#ifndef LIBTENSOR_EXPR_NODE_SET_H
#define LIBTENSOR_EXPR_NODE_SET_H

#include "node_product.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: set elements of an expression to a value

    This expression node copies a tensor or subexpression and sets elements to
    a constant value. The node has two arguments: the subexpression and a
    scalar. The elements which are set to the value are defined by the array
    of tensor indices whose length must be the rank of the input tensor.
    Identical values in the tensor indices define dimensions for which the
    generalized diagonal is set to the value. The order of dimensions in the
    output is identical to the input.

    Examples:
    A(ij):    I = ( 0 1 )
    A(ii):    I = ( 0 0 )
    A(iaib):  I = ( 0 1 0 2 )
    A(iiaca): I = ( 0 0 1 2 1 )

    \ingroup libtensor_expr
 **/
class node_set : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::vector<size_t> m_idx; //!< Tensor indices
    bool m_add; //!< Add to

public:
    /** \brief Creates a node
        \param idx Tensor indices.
        \param add If addition is requested
     **/
    node_set(const std::vector<size_t> &idx, bool add = false) :
        node(node_set::k_op_type, idx.size()), m_idx(idx), m_add(add)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_set() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_set(*this);
    }

    /** \brief Return the diagonal mask
     **/
    const std::vector<size_t> &get_idx() const {
        return m_idx;
    }

    /** \brief Return if addition is requested
     **/
    bool add() const { return m_add; }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SET_H
