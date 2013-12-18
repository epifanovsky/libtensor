#ifndef LIBTENSOR_EXPR_NODE_DIAG_H
#define LIBTENSOR_EXPR_NODE_DIAG_H

#include <vector>
#include "node_product.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: extraction of a general diagonal

    This expression node represent the extraction of a general diagonal from
    a tensor or subexpression, which a single argument to this operation.

    See node_product for a full description of the data structure.

    The diagonal index is the repeating index the diagonal to be extracted.
    For example, A(ii) has indices (0, 0) with the diagonal index 0;
    A(aii) has indices (0, 1, 1) with the diagonal index 1.

    \ingroup libtensor_expr
 **/
class node_diag : public node_product {
public:
    static const char k_op_type[]; //!< Operation type

private:
    size_t m_didx; //!< Diagonal index

public:
    /** \brief Creates an identity node
        \param n Order of result.
        \param idx Tensor indices.
        \param didx Index of the diagonal.
     **/
    node_diag(size_t n, const std::vector<size_t> &idx, size_t didx) :
        node_product(node_diag::k_op_type, n, idx), m_didx(didx)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_diag() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_diag *clone() const {
        return new node_diag(*this);
    }

    /** \brief Return the index on the diagonal
     **/
    size_t get_didx() const {
        return m_didx;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIAG_H
