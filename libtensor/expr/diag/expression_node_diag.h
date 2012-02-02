#ifndef LIBTENSOR_EXPRESSION_NODE_DIAG_H
#define LIBTENSOR_EXPRESSION_NODE_DIAG_H

#include <cstring>
#include <libtensor/core/permutation.h>
#include "../anytensor.h"
#include "../expression.h"

namespace libtensor {


/** \brief Expression node: general diagonal of a tensor (base class)

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class expression_node_diag_base : public expression_node<N, T> {
public:
    static const char *k_node_type; //!< Node type

public:
    /** \brief Virtual destructor
     **/
    virtual ~expression_node_diag_base() { }

    /** \brief Returns the type of the node
     **/
    virtual const char *get_type() const {
        return k_node_type;
    }

    /** \brief Returns the order of the tensor argument
     **/
    virtual size_t get_n() const = 0;

    /** \brief Returns the order of the diagonal
     **/
    virtual size_t get_m() const = 0;

public:
    /** \brief Returns true if the type of a node is
            expression_node_dirsum_base
     **/
    static bool check_type(expression_node<N, T> &n) {
        return ::strcmp(n.get_type(), k_node_type) == 0;
    }

    /** \brief Casts an abstract node as expression_node_diag
     **/
    static expression_node_diag_base<N, T> &cast(
        expression_node<N, T> &n) {

        return dynamic_cast<expression_node_diag_base<N, T>&>(n);
    }

};


/** \brief Expression node: general diagonal of a tensor
    \tparam N Tensor order.
    \tparam M Order of diagonal.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
class expression_node_diag : public expression_node_diag_base<N - M + 1, T> {
private:
    expression<N, T> m_a; //!< Argument
    mask<N> m_msk; //!< Mask of general diagonal
    permutation<N - M + 1> m_perm; //!< Permutation of result
    T m_s; //!< Scaling coefficient

public:
    /** \brief Initializes the node
     **/
    expression_node_diag(expression<N, T> &a, const mask<N> &msk,
        const permutation<N - M + 1> &perm) :
        m_a(a), m_msk(msk), m_perm(perm), m_s(1) { }

    /** \brief Virtual destructor
     **/
    virtual ~expression_node_diag() { }

    /** \brief Clones the node
     **/
    virtual expression_node<N - M + 1, T> *clone() const {
        return new expression_node_diag<N, M, T>(*this);
    }

    /** \brief Applies a scaling coefficient
     **/
    virtual void scale(const T &s) {
        m_s = m_s * s;
    }

    /** \brief Returns the order of the tensor
     **/
    virtual size_t get_n() const {
        return N;
    }

    /** \brief Returns the order of the diagonal
     **/
    virtual size_t get_m() const {
        return M;
    }

    /** \brief Returns the first argument
     **/
    const expression<N, T> &get_a() const {
        return m_a;
    }

    /** \brief Returns the mask
     **/
    const permutation<N> &get_mask() const {
        return m_msk;
    }

    /** \brief Returns the permutation
     **/
    const permutation<N - M + 1> &get_perm() const {
        return m_perm;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_s() const {
        return m_s;
    }

public:
    /** \brief Casts an abstract node as expression_node_dirsum
     **/
    static expression_node_diag<N, M, T> &cast(
        expression_node<N - M + 1, T> &n) {

        return dynamic_cast<expression_node_diag<N, M, T>&>(n);
    }

};


template<size_t N, typename T>
const char *expression_node_diag_base<N, T>::k_node_type = "diag";


} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_NODE_DIAG_H
