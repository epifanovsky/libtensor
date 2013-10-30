#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H

#include <typeinfo>
#include <vector>
#include "unary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node base class

    \ingroup libtensor_expr
 **/
class node_transform_base : public unary_node_base {
private:
    std::vector<size_t> m_perm; //!< Permutation of indices

public:
    /** \brief Creates a transformation node with index permutation
        \param node Node argument.
        \param perm Permutation of indices.
     **/
    node_transform_base(
        const node &node,
        const std::vector<size_t> &perm) :

        unary_node_base("transform", node), m_perm(perm) {

        check();
    }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the type of the tensor element
     **/
    virtual const std::type_info &get_type() const = 0;

    /** \brief Returns the permutation of tensor indices
     **/
    const std::vector<size_t> &get_perm() const {
        return m_perm;
    }

private:
    void check();

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
