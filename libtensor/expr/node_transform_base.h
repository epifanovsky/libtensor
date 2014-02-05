#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H

#include <typeinfo>
#include <vector>
#include <libtensor/exception.h>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor transformation node base class

    \ingroup libtensor_expr
 **/
class node_transform_base : public node {
public:
    static const char k_op_type[];

private:
    std::vector<size_t> m_perm; //!< Permutation of indices

public:
    /** \brief Creates a transformation node with index permutation
        \param perm Permutation of indices.
     **/
    node_transform_base(const std::vector<size_t> &perm) :
        node(node_transform_base::k_op_type, perm.size()), m_perm(perm)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the permutation of tensor indices
     **/
    const std::vector<size_t> &get_perm() const {
        return m_perm;
    }

    /** \brief Returns the scalar type of transformation
     **/
    virtual const std::type_info &get_type() const = 0;

private:
    void check() {
#ifdef LIBTENSOR_DEBUG
    std::vector<bool> ok(m_perm.size(), false);
    for(size_t i = 0; i < m_perm.size(); i++) {
        if (ok[m_perm[i]]) {
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index duplicate.");
        }

        ok[m_perm[i]] = true;
    }

    for(size_t i = 0; i < m_perm.size(); i++) {
        if(!ok[m_perm[i]])
            throw generic_exception(g_ns, "node_transform_base", "check()",
                    __FILE__, __LINE__, "Index missing.");
    }
#endif // LIBTENSOR_DEBUG
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_TRANSFORM_BASE_H
