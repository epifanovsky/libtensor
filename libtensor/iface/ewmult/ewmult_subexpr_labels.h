#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABELS_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABELS_H

#include "ewmult_core.h"
#include "ewmult_subexpr_label_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Labels for sub-expressions in ewmult

    Each contract expression has three parameters: two sub-expressions
    for the contraction arguments and a label that specifies the letter
    indexes over which the contraction is to be performed.

    Since the sub-expressions are arbitrary expressions themselves, they
    may have to be evaluated into temporary tensors. This class selects
    the result labels for the evaluation of the sub-expressions such that
    the overall computation time is minimized.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class ewmult_subexpr_labels {
public:
    static const size_t k_ordera = N + K; //!< Order of the first %tensor
    static const size_t k_orderb = M + K; //!< Order of the second %tensor
    static const size_t k_orderc = N + M + K; //!< Order of the result

    //!    Element-wise product expression core type
    typedef ewmult_core<N, M, K, T, E1, E2> core_t;

    //!    Contraction expression type
    typedef expr<k_orderc, T, core_t> expression_t;

private:
    //!    Label builder for the first sub-expression
    ewmult_subexpr_label_builder<N, M, K> m_bld_a;

    //!    Label builder for the second sub-expression
    ewmult_subexpr_label_builder<M, N, K> m_bld_b;

public:
    /** \brief Initializes the object using an ewmult expression and
            a result label
     **/
    ewmult_subexpr_labels(
        const expression_t &expr, const letter_expr<k_orderc> &label_c);

    /** \brief Returns the label for the first sub-expression
     **/
    const letter_expr<N + K> &get_label_a() const;

    /** \brief Returns the label for the second sub-expression
     **/
    const letter_expr<M + K> &get_label_b() const;
};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
ewmult_subexpr_labels<N, M, K, T, E1, E2>::ewmult_subexpr_labels(
    const expression_t &expr, const letter_expr<k_orderc> &label_c) :

    m_bld_a(label_c, expr.get_core().get_ewidx(),
        expr.get_core().get_expr_1()),
    m_bld_b(label_c, expr.get_core().get_ewidx(),
        expr.get_core().get_expr_2()) {

}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline const letter_expr<N + K> &ewmult_subexpr_labels<N, M, K, T,
E1, E2>::get_label_a() const {

    return m_bld_a.get_label();
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline const letter_expr<M + K> &ewmult_subexpr_labels<N, M, K, T,
E1, E2>::get_label_b() const {

    return m_bld_b.get_label();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABELS_H
