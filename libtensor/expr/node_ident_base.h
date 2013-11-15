#ifndef LIBTENSOR_EXPR_NODE_IDENT_BASE_H
#define LIBTENSOR_EXPR_NODE_IDENT_BASE_H

#include <libtensor/iface/any_tensor.h>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Base class for tensor identity node of the expression tree

    \ingroup libtensor_expr
 **/
class node_ident_base : public node {
public:
    static const char k_op_type[]; //!< Operation type

public:
    /** \brief Constructor
        \param n Tensor order
     **/
    node_ident_base(size_t n) :
        node(node_ident_base::k_op_type, n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ident_base() { }

    virtual const std::type_info &get_t() const = 0;

    /** \brief Checks if both identity nodes contain the same tensor
     **/
    virtual bool operator==(const node_ident_base &n) const = 0;
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_IDENT_H
