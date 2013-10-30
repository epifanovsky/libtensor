#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_H

#include <libtensor/core/scalar_transf.h>
#include "node_transform_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node class (double)

    \ingroup libtensor_expr
 **/
template<typename T>
class node_transform : public node_transform_base {
private:
    scalar_transf<T> m_coeff; //!< Scaling coefficient

public:
    /** \brief Creates a transformation node
        \param node Child node.
        \param perm Permutation of indices.
        \param c Scaling coefficient.
     **/
    node_transform(
        const node &node,
        const std::vector<size_t> &perm,
        const scalar_transf<T> &c) :

        node_transform_base(node, perm), m_coeff(c)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_transform<T> *clone() const {
        return new node_transform<T>(*this);
    }

    /** \brief Returns the type of the tensor element
     **/
    virtual const std::type_info &get_type() const {
        return typeid(double);
    }

    /** \brief Returns the scaling coefficient
     **/
    const scalar_transf<T> &get_coeff() const {
        return m_coeff;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_H
