#ifndef LIBTENSOR_EXPRESSION_NODE_IDENT_H
#define LIBTENSOR_EXPRESSION_NODE_IDENT_H

#include <cstring>
#include <libtensor/core/permutation.h>
#include "../anytensor.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: identity

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_node_ident : public expression_node<N, T> {
public:
    static const char *k_node_type; //!< Node type

private:
    anytensor<N, T> &m_t; //!< Tensor
    permutation<N> m_perm; //!< Permutation
    T m_s; //!< Scaling coefficient

public:
    /** \brief Initializes the node
     **/
    expression_node_ident(anytensor<N, T> &t, const permutation<N> &perm) :
        m_t(t), m_perm(perm), m_s(1) { }

    /** \brief Virtual destructor
     **/
    virtual ~expression_node_ident() { }

    /** \brief Clones the node
     **/
    virtual expression_node<N, T> *clone() const {
        return new expression_node_ident<N, T>(*this);
    }

    /** \brief Returns the type of the node
     **/
    virtual const char *get_type() const {
        return k_node_type;
    }

    /** \brief Applies a scaling coefficient
     **/
    virtual void scale(const T &s) {
        m_s = m_s * s;
    }

    /** \brief Returns the tensor
     **/
    anytensor<N, T> &get_tensor() {
        return m_t;
    }

    /** \brief Returns the permutation
     **/
    const permutation<N> &get_perm() const {
        return m_perm;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_s() const {
        return m_s;
    }

public:
    /** \brief Returns true if the type of a node is expression_node_ident
     **/
    static bool check_type(expression_node<N, T> &n) {
        return ::strcmp(n.get_type(), k_node_type) == 0;
    }

    /** \brief Casts an abstract node as expression_node_ident
     **/
    static expression_node_ident<N, T> &cast(expression_node<N, T> &n) {
        return dynamic_cast<expression_node_ident<N, T>&>(n);
    }

};


template<size_t N, typename T>
const char *expression_node_ident<N, T>::k_node_type = "ident";


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_NODE_IDENT_H
