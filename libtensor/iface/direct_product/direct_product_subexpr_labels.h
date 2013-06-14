#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABELS_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABELS_H

#include "direct_product_subexpr_label_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Labels for sub-expressions in the direct product

    Each direct product expression has two arbitrary sub-expressions as
    parameters. Before computing the direct product, the expressions may
    have to be evaluated into temporary tensors. This class selects the
    result labels for this evaluation such that the overall computation
    time is minimized.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T>
class direct_product_subexpr_labels {
public:
    enum {
        NA = N,
        NB = M,
        NC = N + M
    };

private:
    //!    Label builder for the first sub-expression
    direct_product_subexpr_label_builder<N, M> m_bld_a;

    //!    Label builder for the second sub-expression
    direct_product_subexpr_label_builder<M, N> m_bld_b;

public:
    /** \brief Initializes the object using a direct product expression
            and a result label
     **/
    direct_product_subexpr_labels(
        const direct_product_core<N, M, T> &core,
        const letter_expr<NC> &label_c);

    /** \brief Returns the label for the first sub-expression
     **/
    const letter_expr<N> &get_label_a() const;

    /** \brief Returns the label for the second sub-expression
     **/
    const letter_expr<M> &get_label_b() const;
};


template<size_t N, size_t M, typename T>
direct_product_subexpr_labels<N, M, T>::direct_product_subexpr_labels(
    const direct_product_core<N, M, T> &core,
    const letter_expr<NC> &label_c) :

    m_bld_a(label_c, core.get_expr_1()),
    m_bld_b(label_c, core.get_expr_2()) {

}


template<size_t N, size_t M, typename T>
const letter_expr<N>&
direct_product_subexpr_labels<N, M, T>::get_label_a() const {

    return m_bld_a.get_label();
}


template<size_t N, size_t M, typename T>
const letter_expr<M>&
direct_product_subexpr_labels<N, M, T>::get_label_b() const {

    return m_bld_b.get_label();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABELS_H
