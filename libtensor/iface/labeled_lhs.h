#ifndef LIBTENSOR_IFACE_LABELED_LHS_H
#define LIBTENSOR_IFACE_LABELED_LHS_H

#include "expr_lhs.h"
#include "labeled.h"

namespace libtensor {
namespace iface {


/** \brief Labeled left-hand-side of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_lhs : virtual public labeled<N> {
private:
    expr_lhs<N, T> &m_lhs; //!< Left-hand-side
    letter_expr<N> m_label; //!< Letter label

public:
    /** \brief Initializes the labeled LHS
     **/
    labeled_lhs(expr_lhs<N, T> &lhs, const letter_label<N> &label) :
        m_lhs(lhs), m_label(label)
    { }

    /** \brief Assignment of the right-hand-side of a tensor expression to
            the left-hand-side
     **/
    expr_rhs<N, T> &operator=(const expr_rhs<N, T> &rhs) {
        m_lhs.assign(rhs, m_label);
        return rhs;
    }

};


/** \brief Attaches a letter label to the left-hand-side of a tensor expression
 **/
template<size_t N, typename T>
labeled_lhs<N, T> operator()(expr_lhs<N, T> &lhs, const letter_expr<N> &label) {

    return labeled_lhs<N, T>(lhs, label);
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_LABELED_LHS_H
