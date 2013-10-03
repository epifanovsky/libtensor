#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H

#include "node_transform_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node class


    \ingroup libtensor_expr
 **/
class node_transform_double : public node_transform_base {
private:
    double m_coeff; //!< Scaling coefficient

public:
    /** \brief Creates an identity node
        \param node Node argument.
        \param order Index order.
        \param c Scaling coefficient.
     **/
    node_transform_double(const node &node,
            const std::vector<size_t> &order, double c) :
        node_transform_base("double", node, order), m_coeff(c)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_double() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_transform_double *clone() const {
        return new node_transform_double(*this);
    }

    const double &get_coeff() const {
        return m_coeff;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H
