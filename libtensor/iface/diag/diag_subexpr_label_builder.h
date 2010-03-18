#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_SUBEXPR_LABEL_BUILDER_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename Core> class expr;


/**	\brief Label builder for sub-expressions in diag

	\ingroup libtensor_iface
 **/
template<size_t N, size_t M>
class diag_subexpr_label_builder {
private:
	struct letter_array {
	private:
		const letter *m_let[N];
	public:
		template<typename T, typename Core>
		letter_array(const letter_expr<N - M + 1> &label_b,
			const letter &letter_diag,
			const letter_expr<M> &label_diag,
			const expr<N, T, Core> &e);
		const letter *at(size_t i) const { return m_let[i]; }
	};
	template<size_t L>
	struct dummy { };
	letter_array m_let;
	letter_expr<N> m_label;

public:
	template<typename T, typename Core>
	diag_subexpr_label_builder(const letter_expr<N - M + 1> &label_b,
		const letter &letter_diag, const letter_expr<M> &label_diag,
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
diag_subexpr_label_builder<N, M>::diag_subexpr_label_builder(
	const letter_expr<N - M + 1> &label_b, const letter &letter_diag,
	const letter_expr<M> &label_diag, const expr<N, T, Core> &e) :

	m_let(label_b, letter_diag, label_diag, e),
	m_label(mk_label(dummy<N>(), m_let, N - 1)) {

}


template<size_t N, size_t M> template<typename T, typename Core>
diag_subexpr_label_builder<N, M>::letter_array::letter_array(
	const letter_expr<N - M + 1> &label_b, const letter &letter_diag,
	const letter_expr<M> &label_diag, const expr<N, T, Core> &e) {

	//	We assume here that all consistency checks have been done.

	size_t j = 0;
	for(size_t i = 0; i < N - M + 1; i++) {
		const letter &l = label_b.letter_at(i);
		if(l.equals(letter_diag)) {
			for(size_t ii = 0; ii < M; ii++) {
				m_let[j++] = &label_diag.letter_at(ii);
			}
		} else {
			m_let[j++] = &l;
		}

	}
}


template<size_t N, size_t M> template<size_t L>
letter_expr<L> diag_subexpr_label_builder<N, M>::mk_label(
	const dummy<L>&, const letter_array &let, size_t i) {

	letter_expr<L - 1> sublabel = mk_label(dummy<L - 1>(), let, i - 1);
	return letter_expr<L>(sublabel, *let.at(i));
}


template<size_t N, size_t M>
letter_expr<1> diag_subexpr_label_builder<N, M>::mk_label(
	const dummy<1>&, const letter_array &let, size_t i) {

	return letter_expr<1>(*let.at(i));
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_SUBEXPR_LABEL_BUILDER_H
