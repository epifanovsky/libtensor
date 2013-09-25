#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABEL_BUILDER_H

#include <libtensor/core/sequence.h>

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Label builder for sub-expressions in direct products

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M>
class direct_product_subexpr_label_builder {
private:
    struct letter_array {
    private:
        sequence<N, const letter*> m_let;
    public:
        template<typename T>
        letter_array(
            const letter_expr<N + M> &label_c,
            const expr_rhs<N, T> &e);
        const letter *at(size_t i) const { return m_let[i]; }
    };
    template<size_t L>
    struct dummy { };
    letter_array m_let;
    letter_expr<N> m_label;

public:
    template<typename T>
    direct_product_subexpr_label_builder(
        const letter_expr<N + M> &label_c,
        const expr_rhs<N, T> &e);

    const letter_expr<N> &get_label() const {
        return m_label;
    }

protected:
    template<size_t L>
    static letter_expr<L> mk_label(
        const dummy<L>&, const letter_array &let, size_t i);
    static letter_expr<1> mk_label(
        const dummy<1>&, const letter_array &let, size_t i);

};


template<size_t N, size_t M>
template<typename T>
direct_product_subexpr_label_builder<N, M>::direct_product_subexpr_label_builder(
    const letter_expr<N + M> &label_c, const expr_rhs<N, T> &e) :

    m_let(label_c, e),
    m_label(mk_label(dummy<N>(), m_let, N - 1)) {

}


template<size_t N, size_t M>
template<typename T>
direct_product_subexpr_label_builder<N, M>::letter_array::letter_array(
    const letter_expr<N + M> &label_c, const expr_rhs<N, T> &e) :

    m_let(0) {

    size_t j = 0;
    for(size_t i = 0; i < N + M; i++) {
        const letter &l = label_c.letter_at(i);
        if(e.get_core().contains(l)) {
            if(j == N) {
                throw_exc("direct_product_subexpr_label_builder::letter_array",
                    "letter_array()", "Inconsistent expression");
            }
            m_let[j++] = &l;
        }
    }
    if(j != N) {
        throw_exc("direct_product_subexpr_label_builder::letter_array",
            "letter_array()", "Inconsistent expression");
    }
}


template<size_t N, size_t M>
template<size_t L>
letter_expr<L> direct_product_subexpr_label_builder<N, M>::mk_label(
    const dummy<L>&, const letter_array &let, size_t i) {

    letter_expr<L - 1> sublabel = mk_label(dummy<L - 1>(), let, i - 1);
    return letter_expr<L>(sublabel, *let.at(i));
}


template<size_t N, size_t M>
letter_expr<1> direct_product_subexpr_label_builder<N, M>::mk_label(
    const dummy<1>&, const letter_array &let, size_t i) {

    return letter_expr<1>(*let.at(i));
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_SUBEXPR_LABEL_BUILDER_H
