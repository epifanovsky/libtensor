#ifndef LIBTENSOR_EXPR_NODE_TRANSFORM_H
#define LIBTENSOR_EXPR_NODE_TRANSFORM_H

#include <typeinfo>
#include <vector>
#include <libtensor/exception.h>
#include <libtensor/core/scalar_transf.h>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: tensor transformation (base class)

    \ingroup libtensor_expr_dag
 **/
class node_transform_base : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::vector<size_t> m_perm; //!< Permutation of indices

public:
    /** \brief Creates a transformation node with index permutation
        \param perm Permutation of indices.
     **/
    node_transform_base(const std::vector<size_t> &perm) :
        node(k_op_type, perm.size()), m_perm(perm)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform_base() { }

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


/** \brief Expression node: tensor transformation

    \ingroup libtensor_expr_dag
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
        const std::vector<size_t> &perm,
        const scalar_transf<T> &c) :

        node_transform_base(perm), m_coeff(c)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_transform() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_transform<T>(*this);
    }

    /** \brief Returns the scalar type of transformation
     **/
    virtual const std::type_info &get_type() const {
        return typeid(T);
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
