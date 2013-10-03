#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H

#include "node_transform_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node class (double)

    \ingroup libtensor_expr
 **/
class node_transform_double : public node_transform_base {
private:
    double m_coeff; //!< Scaling coefficient

public:
    /** \brief Creates a transformation node
        \param node Child node.
        \param perm Permutation of indices.
        \param c Scaling coefficient.
     **/
    node_transform_double(const node &node, const std::vector<size_t> &perm,
        double c) :
        node_transform_base(node, perm), m_coeff(c)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_double() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_transform_double(*this);
    }

    /** \brief Returns the type of the tensor element
     **/
    virtual const std::type_info &get_type() const {
        return typeid(double);
    }

    /** \brief Returns the scaling coefficient
     **/
    double get_coeff() const {
        return m_coeff;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_DOUBLE_H
