#ifndef LIBTENSOR_EXPRESSION_NODE_DIRPROD_H
#define LIBTENSOR_EXPRESSION_NODE_DIRPROD_H

#include <cstring>
#include <libtensor/core/permutation.h>
#include "../anytensor.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: direct product of two tensors (base class)

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_node_dirprod_base : public expression_node<N, T> {
public:
    static const char *k_node_type; //!< Node type

public:
    /** \brief Virtual destructor
     **/
    virtual ~expression_node_dirprod_base() { }

    /** \brief Returns the type of the node
     **/
    virtual const char *get_type() const {
        return k_node_type;
    }

    /** \brief Returns the order of the first tensor
     **/
    virtual size_t get_n() const = 0;

    /** \brief Returns the order of the second tensor
     **/
    virtual size_t get_m() const = 0;

public:
    /** \brief Returns true if the type of a node is
            expression_node_dirprod_base
     **/
    static bool check_type(expression_node<N, T> &n) {
        return ::strcmp(n.get_type(), k_node_type) == 0;
    }

    /** \brief Casts an abstract node as expression_node_dirprod
     **/
    static expression_node_dirprod_base<N, T> &cast(
        expression_node<N, T> &n) {

        return dynamic_cast<expression_node_dirprod_base<N, T>&>(n);
    }

};


/** \brief Expression node: direct product of two tensors

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class expression_node_dirprod : public expression_node_dirprod_base<N + M, T> {
private:
    expression<N, T> m_a; //!< First argument
    expression<M, T> m_b; //!< Second argument
    permutation<N + M> m_perm; //!< Permutation
    T m_s; //!< Scaling coefficient

public:
    /** \brief Initializes the node
     **/
    expression_node_dirprod(expression<N, T> &a, expression<M, T> &b,
        const permutation<N + M> &perm) :
        m_a(a), m_b(b), m_perm(perm), m_s(1) { }

    /** \brief Virtual destructor
     **/
    virtual ~expression_node_dirprod() { }

    /** \brief Clones the node
     **/
    virtual expression_node<N + M, T> *clone() const {
        return new expression_node_dirprod<N, M, T>(*this);
    }

    /** \brief Applies a scaling coefficient
     **/
    virtual void scale(const T &s) {
        m_s = m_s * s;
    }

    /** \brief Returns the order of the first tensor
     **/
    virtual size_t get_n() const {
        return N;
    }

    /** \brief Returns the order of the second tensor
     **/
    virtual size_t get_m() const {
        return M;
    }

    /** \brief Returns the first argument
     **/
    const expression<N, T> &get_a() const {
        return m_a;
    }

    /** \brief Returns the second argument
     **/
    const expression<M, T> &get_b() const {
        return m_b;
    }

    /** \brief Returns the permutation
     **/
    const permutation<N + M> &get_perm() const {
        return m_perm;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_s() const {
        return m_s;
    }

public:
    /** \brief Casts an abstract node as expression_node_dirprod
     **/
    static expression_node_dirprod<N, M, T> &cast(
        expression_node<N + M, T> &n) {

        return dynamic_cast<expression_node_dirprod<N, M, T>&>(n);
    }

};


template<size_t N, typename T>
const char *expression_node_dirprod_base<N, T>::k_node_type = "dirprod";


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_NODE_DIRPROD_H
