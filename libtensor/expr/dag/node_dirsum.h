#ifndef LIBTENSOR_EXPR_NODE_DIRSUM_H
#define LIBTENSOR_EXPR_NODE_DIRSUM_H

#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor expression node: direct sum of two tensors

    \ingroup libtensor_expr_dag
 **/
class node_dirsum : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Creates an direct tensor sum node of two tensors
        \param n Tensor order of the result.
     **/
    node_dirsum(size_t n) :
        node(k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_dirsum() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_dirsum(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_DIRSUM_H
