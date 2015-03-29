#ifndef LIBTENSOR_EXPR_EXPR_LHS_H
#define LIBTENSOR_EXPR_EXPR_LHS_H

#include "expr_rhs.h"

namespace libtensor {
namespace expr {


template<size_t N, typename T> class labeled_lhs;


/** \brief Assignable left-hand side of a tensor expression

    \ingroup libtensor_expr_iface
 **/
template<size_t N, typename T>
class expr_lhs {
public:
    /** \brief Destructor
     **/
    virtual ~expr_lhs() { }

    /** \brief Performs the assignment of the right-hand side to this
            left-hand side of the expression
     **/
    virtual void assign(const expr_rhs<N, T> &rhs, const label<N> &l) = 0;

    /** \brief Performs the assignment with addition of the right-hand side
            to this left-hand side of the expression
     **/
    virtual void assign_add(const expr_rhs<N, T> &rhs, const label<N> &l) = 0;

    /** \brief Attaches a letter label to the left-hand-side of a tensor
            expression
     **/
    labeled_lhs<N, T> operator()(const label<N> &l);

};


/** \brief Labeled left-hand-side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_lhs {
private:
    expr_lhs<N, T> &m_lhs; //!< Left-hand-side
    label<N> m_label; //!< Letter label

public:
    /** \brief Initializes the labeled LHS
     **/
    labeled_lhs(expr_lhs<N, T> &lhs, const label<N> &l) :
        m_lhs(lhs), m_label(l)
    { }

    /** \brief Assignment of the right-hand-side of a tensor expression to
            the left-hand-side
     **/
    const expr_rhs<N, T> &operator=(const expr_rhs<N, T> &rhs);

    /** \brief Assignment with addition of the right-hand-side of a tensor
            expression to the left-hand-side
     **/
    const expr_rhs<N, T> &operator+=(const expr_rhs<N, T> &rhs);

};


template<size_t N, typename T>
labeled_lhs<N, T> expr_lhs<N, T>::operator()(const label<N> &l) {

    return labeled_lhs<N, T>(*this, l);
}


template<size_t N, typename T>
const expr_rhs<N, T> &labeled_lhs<N, T>::operator=(const expr_rhs<N, T> &rhs) {

    m_lhs.assign(rhs, m_label);
    return rhs;
}


template<size_t N, typename T>
const expr_rhs<N, T> &labeled_lhs<N, T>::operator+=(const expr_rhs<N, T> &rhs) {

    m_lhs.assign_add(rhs, m_label);
    return rhs;
}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EXPR_LHS_H
