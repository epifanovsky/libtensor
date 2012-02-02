#ifndef LIBTENSOR_EXPRESSION_NODE_CONTRACT2_H
#define LIBTENSOR_EXPRESSION_NODE_CONTRACT2_H

#include <cstring>
#include <libtensor/tod/contraction2.h>
#include "../anytensor.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: contraction of two tensors (base class)

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_node_contract2_base : public expression_node<N, T> {
public:
    static const char *k_node_type; //!< Node type

public:
    /** \brief Virtual destructor
     **/
    virtual ~expression_node_contract2_base() { }

    /** \brief Returns the type of the node
     **/
    virtual const char *get_type() const {
        return k_node_type;
    }

    /** \brief Returns the order of the first tensor less contraction degree
     **/
    virtual size_t get_n() const = 0;

    /** \brief Returns the order of the second tensor less contraction degree
     **/
    virtual size_t get_m() const = 0;

    /** \brief Returns the number of contracted indexes
     **/
    virtual size_t get_k() const = 0;

public:
    /** \brief Returns true if the type of a node is
            expression_node_contract2_base
     **/
    static bool check_type(expression_node<N, T> &n) {
        return ::strcmp(n.get_type(), k_node_type) == 0;
    }

    /** \brief Casts an abstract node as expression_node_contract2_base
     **/
    static expression_node_contract2_base<N, T> &cast(
        expression_node<N, T> &n) {

        return dynamic_cast<expression_node_contract2_base<N, T>&>(n);
    }

};


/** \brief Expression node: contraction of two tensors

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T>
class expression_node_contract2 :
    public expression_node_contract2_base<N + M, T> {

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    expression<N + K, T> m_a; //!< First argument
    expression<M + K, T> m_b; //!< Second argument
    T m_s; //!< Scaling coefficient

public:
    /** \brief Initializes the node
     **/
    expression_node_contract2(const contraction2<N, M, K> &contr,
        expression<N + K, T> &a, expression<M + K, T> &b) :
        m_contr(contr), m_a(a), m_b(b), m_s(1) { }

    /** \brief Virtual destructor
     **/
    virtual ~expression_node_contract2() { }

    /** \brief Clones the node
     **/
    virtual expression_node<N + M, T> *clone() const {
        return new expression_node_contract2<N, M, K, T>(*this);
    }

    /** \brief Applies a scaling coefficient
     **/
    virtual void scale(const T &s) {
        m_s = m_s * s;
    }

    /** \brief Returns the order of the first tensor less contraction degree
     **/
    virtual size_t get_n() const {
        return N;
    }

    /** \brief Returns the order of the second tensor less contraction degree
     **/
    virtual size_t get_m() const {
        return M;
    }

    /** \brief Returns the number of contracted indexes
     **/
    virtual size_t get_k() const {
        return K;
    }

    /** \brief Returns the contraction
     **/
    const contraction2<N, M, K> &get_contr() const {
        return m_contr;
    }

    /** \brief Returns the first argument
     **/
    const expression<N + K, T> &get_a() const {
        return m_a;
    }

    /** \brief Returns the second argument
     **/
    const expression<M + K, T> &get_b() const {
        return m_b;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_s() const {
        return m_s;
    }

public:
    /** \brief Casts an abstract node as expression_node_contract2
     **/
    static expression_node_contract2<N, M, K, T> &cast(
        expression_node<N + M, T> &n) {

        return dynamic_cast<expression_node_contract2<N, M, K, T>&>(n);
    }

};


template<size_t N, typename T>
const char *expression_node_contract2_base<N, T>::k_node_type = "contract2";


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_NODE_CONTRACT2_H
