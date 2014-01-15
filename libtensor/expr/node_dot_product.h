#ifndef LIBTENSOR_EXPR_NODE_DOT_PRODUCT_H
#define LIBTENSOR_EXPR_NODE_DOT_PRODUCT_H

#include <vector>
#include "node_product.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: dot product of two tensors

    This expression node represent the dot product between two tensors or
    tensor subexpressions. The result is a scalar.

    See node_product for a full description of the data structure.

    The diagonal index is the repeating index the diagonal to be extracted.
    For example, A(ii) has indices (0, 0) with the diagonal index 0;
    A(aii) has indices (0, 1, 1) with the diagonal index 1.

    \ingroup libtensor_expr
 **/
class node_dot_product : public node_product {
public:
    static const char k_clazz[]; //!< Class name
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates the node
        \param idxa Indices of first tensor.
        \param idxb Indices of second tensor.
     **/
    node_dot_product(
        const std::vector<size_t> &idxa,
        const std::vector<size_t> &idxb) :
        node_product(k_op_type, 0, make_idx(idxa, idxb), make_cidx(idxa, idxb))
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_dot_product() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_dot_product(*this);
    }

private:
    static std::vector<size_t> make_idx(
        const std::vector<size_t> &idxa, const std::vector<size_t> &idxb);

    static std::vector<size_t> make_cidx(
        const std::vector<size_t> &idxa, const std::vector<size_t> &idxb);

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DOT_PRODUCT_H
