#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABEL_BUILDER_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename Core> class expr;


/** \brief Label builder for sub-expressions in dirsum

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M>
class dirsum_subexpr_label_builder {
private:
    struct letter_array {
    private:
        const letter *m_let[N];
    public:
        template<typename T, typename Core>
        letter_array(const letter_expr<N + M> &label_c,
            const expr<N, T, Core> &e);
        const letter *at(size_t i) const { return m_let[i]; }
    };
    template<size_t L>
    struct dummy { };
    letter_array m_let;
    letter_expr<N> m_label;

public:
    template<typename T, typename Core>
    dirsum_subexpr_label_builder(const letter_expr<N + M> &label_c,
        const expr<N, T, Core> &e);

    const letter_expr<N> &get_label() const { return m_label; }

protected:
    template<size_t L>
    static letter_expr<L> mk_label(
        const dummy<L>&, const letter_array &let, size_t i);
    static letter_expr<1> mk_label(
        const dummy<1>&, const letter_array &let, size_t i);

};


template<size_t N, size_t M> template<typename T, typename Core>
dirsum_subexpr_label_builder<N, M>::dirsum_subexpr_label_builder(
    const letter_expr<N + M> &label_c, const expr<N, T, Core> &e) :

    m_let(label_c, e),
    m_label(mk_label(dummy<N>(), m_let, N - 1)) {

}


template<size_t N, size_t M> template<typename T, typename Core>
dirsum_subexpr_label_builder<N, M>::letter_array::letter_array(
    const letter_expr<N + M> &label_c, const expr<N, T, Core> &e) {

    for(size_t i = 0; i < N; i++) m_let[i] = &e.letter_at(i);
}


template<size_t N, size_t M> template<size_t L>
letter_expr<L> dirsum_subexpr_label_builder<N, M>::mk_label(
    const dummy<L>&, const letter_array &let, size_t i) {

    letter_expr<L - 1> sublabel = mk_label(dummy<L - 1>(), let, i - 1);
    return letter_expr<L>(sublabel, *let.at(i));
}


template<size_t N, size_t M>
letter_expr<1> dirsum_subexpr_label_builder<N, M>::mk_label(
    const dummy<1>&, const letter_array &let, size_t i) {

    return letter_expr<1>(*let.at(i));
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABEL_BUILDER_H
