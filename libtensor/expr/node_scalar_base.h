#ifndef LIBTENSOR_EXPR_NODE_SCALAR_BASE_H
#define LIBTENSOR_EXPR_NODE_SCALAR_BASE_H

#include <typeinfo>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: scalar container (base class)

    \ingroup libtensor_expr
 **/
class node_scalar_base : public node {
public:
    static const char k_op_type[];

public:
    /** \brief Creates the node
     **/
    node_scalar_base() : node(k_op_type, 0) { }

    /** \brief Virtual destructor
     **/
    virtual ~node_scalar_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the type of scalar
     **/
    virtual const std::type_info &get_type() const = 0;

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SCALAR_BASE_H
