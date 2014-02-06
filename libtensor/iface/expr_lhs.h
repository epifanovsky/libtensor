#ifndef LIBTENSOR_IFACE_EXPR_LHS_H
#define LIBTENSOR_IFACE_EXPR_LHS_H

#include "expr_rhs.h"

namespace libtensor {
namespace iface {


template<size_t N, typename T> class labeled_lhs;


/** \brief Assignable left-hand side of a tensor expression

    \ingroup libtensor_iface
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
    virtual void assign(const expr_rhs<N, T> &rhs,
        const letter_expr<N> &label) = 0;

    /** \brief Attaches a letter label to the left-hand-side of a tensor
            expression
     **/
    labeled_lhs<N, T> operator()(const letter_expr<N> &label);

};


/** \brief Labeled left-hand-side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_lhs {
private:
    expr_lhs<N, T> &m_lhs; //!< Left-hand-side
    letter_expr<N> m_label; //!< Letter label

public:
    /** \brief Initializes the labeled LHS
     **/
    labeled_lhs(expr_lhs<N, T> &lhs, const letter_expr<N> &label) :
        m_lhs(lhs), m_label(label)
    { }

    /** \brief Assignment of the right-hand-side of a tensor expression to
            the left-hand-side
     **/
    const expr_rhs<N, T> &operator=(const expr_rhs<N, T> &rhs);

};


template<size_t N, typename T>
labeled_lhs<N, T> expr_lhs<N, T>::operator()(const letter_expr<N> &label) {

    return labeled_lhs<N, T>(*this, label);
}


template<size_t N, typename T>
const expr_rhs<N, T> &labeled_lhs<N, T>::operator=(const expr_rhs<N, T> &rhs) {

    m_lhs.assign(rhs, m_label);
    return rhs;
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_LHS_H
