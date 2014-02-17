#ifndef LIBTENSOR_EXPR_NODE_TRACE_H
#define LIBTENSOR_EXPR_NODE_TRACE_H

#include "node_product.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: generalized trace of a tensor

    This expression node represent the generalized trace of a tensor or
    an expression. The result is a scalar.

    See node_product for a full description of the data structure.

    \ingroup libtensor_expr
 **/
class node_trace : public node_product {
public:
    static const char k_clazz[]; //!< Class name
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates the node
        \param idx Indices of the tensor tensor.
        \param cidx Contracted indices.
     **/
    node_trace(
        const std::vector<size_t> &idx,
        const std::vector<size_t> &cidx) :
        node_product(k_op_type, 0, idx, cidx)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_trace() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_trace(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRACE_H
