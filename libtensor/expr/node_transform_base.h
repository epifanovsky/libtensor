#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H

#include <vector>
#include <libtensor/exception.h>
#include "unary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node base class

    \ingroup libtensor_expr
 **/
class node_transform_base : public unary_node_base {
private:
    std::string m_type; //!< Transform type
    std::vector<size_t> m_order; //!< Resulting index order (index i -> m_order[i]).
public:
    /** \brief Creates an identity node
        \param node Node argument.
        \param order Index order
     **/
    node_transform_base(const std::string &type, const node &node,
        const std::vector<size_t> &order) :
        unary_node_base("transform", node), m_type(type), m_order(order)
    {
        check();
    }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    const std::string &get_type() const {
        return m_type;
    }

    const std::vector<size_t> &get_order() const {
        return m_order;
    }

private:
    void check() throw(exception);
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
