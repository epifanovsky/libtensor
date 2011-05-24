#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABEL_BUILDER_H

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename Core> class expr;


/**	\brief Label builder for sub-expressions in ewmult

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K>
class ewmult_subexpr_label_builder {
private:
	struct letter_array {
	private:
		sequence<N + K, const letter*> m_let;
	public:
		template<typename T, typename Core>
		letter_array(const letter_expr<N + M + K> &label_c,
			const letter_expr<K> &ewidx,
			const expr<N + K, T, Core> &e);
		const letter *at(size_t i) const { return m_let[i]; }
	};
	template<size_t L>
	struct dummy { };
	letter_array m_let;
	letter_expr<N + K> m_label;

public:
	template<typename T, typename Core>
	ewmult_subexpr_label_builder(const letter_expr<N + M + K> &label_c,
		const letter_expr<K> &ewidx, const expr<N + K, T, Core> &e);

	const letter_expr<N + K> &get_label() const { return m_label; }

protected:
	template<size_t L>
	static letter_expr<L> mk_label(
		const dummy<L>&, const letter_array &let, size_t i);
	static letter_expr<1> mk_label(
		const dummy<1>&, const letter_array &let, size_t i);

};


template<size_t N, size_t M, size_t K> template<typename T, typename Core>
ewmult_subexpr_label_builder<N, M, K>::ewmult_subexpr_label_builder(
	const letter_expr<N + M + K> &label_c, const letter_expr<K> &ewidx,
	const expr<N + K, T, Core> &e) :

	m_let(label_c, ewidx, e),
	m_label(mk_label(dummy<N + K>(), m_let, N + K - 1)) {

}


template<size_t N, size_t M, size_t K> template<typename T, typename Core>
ewmult_subexpr_label_builder<N, M, K>::letter_array::letter_array(
	const letter_expr<N + M + K> &label_c, const letter_expr<K> &ewidx,
	const expr<N + K, T, Core> &e) {

	for(size_t i = 0; i < N + K; i++) m_let[i] = 0;

	size_t j = 0;
	// Take the first indexes from c, not shared (max N)
	for(size_t i = 0; i < N + M + K; i++) {
		const letter &l = label_c.letter_at(i);
		if(e.contains(l) && !ewidx.contains(l)) {
			if(j == N) {
				throw_exc("ewmult_subexpr_label_builder::letter_array",
					"letter_array()", "Inconsistent expression (1)");
			}
			m_let[j++] = &l;
		}
	}
	// Take the shared indexes (max K)
	for(size_t i = 0; i < K; i++) {
		const letter &l = ewidx.letter_at(i);
		if(!e.contains(l)) {
			throw_exc("ewmult_subexpr_label_builder::letter_array",
				"letter_array()", "Inconsistent expression (2)");
		}
		m_let[j++] = &l;
	}
}


template<size_t N, size_t M, size_t K> template<size_t L>
letter_expr<L> ewmult_subexpr_label_builder<N, M, K>::mk_label(
	const dummy<L>&, const letter_array &let, size_t i) {

	letter_expr<L - 1> sublabel = mk_label(dummy<L - 1>(), let, i - 1);
	return letter_expr<L>(sublabel, *let.at(i));
}


template<size_t N, size_t M, size_t K>
letter_expr<1> ewmult_subexpr_label_builder<N, M, K>::mk_label(
	const dummy<1>&, const letter_array &let, size_t i) {

	return letter_expr<1>(*let.at(i));
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_SUBEXPR_LABEL_BUILDER_H
