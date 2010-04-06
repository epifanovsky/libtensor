#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_SUBEXPR_LABEL_BUILDER_H

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Label builder for the sub-expression in trace

	\ingroup libtensor_iface
 **/
template<size_t N>
class trace_subexpr_label_builder {
private:
	struct letter_array {
	private:
		const letter *m_let[2 * N];
	public:
		letter_array(const letter_expr<N> &label1,
			const letter_expr<N> &label1);
		const letter *at(size_t i) const { return m_let[i]; }
	};
	template<size_t L>
	struct dummy { };
	letter_array m_let;
	letter_expr<2 * N> m_label;

public:
	trace_subexpr_label_builder(const letter_expr<N> &label1,
		const letter_expr<N> &label2);

	const letter_expr<2 * N> &get_label() const { return m_label; }

protected:
	template<size_t L>
	static letter_expr<L> mk_label(
		const dummy<L>&, const letter_array &let, size_t i);
	static letter_expr<1> mk_label(
		const dummy<1>&, const letter_array &let, size_t i);

};


template<size_t N>
trace_subexpr_label_builder<N>::trace_subexpr_label_builder(
	const letter_expr<N> &label1, const letter_expr<N> &label2) :

	m_let(label1, label2),
	m_label(mk_label(dummy<2 * N>(), m_let, 2 * N - 1)) {

}


template<size_t N>
trace_subexpr_label_builder<N>::letter_array::letter_array(
	const letter_expr<N> &label1, const letter_expr<N> &label2) {

	for(size_t i = 0; i < N; i++) {
		m_let[i] = &label1.letter_at(i);
		m_let[N + i] = &label2.letter_at(i);
	}
}


template<size_t N> template<size_t L>
letter_expr<L> trace_subexpr_label_builder<N>::mk_label(
	const dummy<L>&, const letter_array &let, size_t i) {

	letter_expr<L - 1> sublabel = mk_label(dummy<L - 1>(), let, i - 1);
	return letter_expr<L>(sublabel, *let.at(i));
}


template<size_t N>
letter_expr<1> trace_subexpr_label_builder<N>::mk_label(
	const dummy<1>&, const letter_array &let, size_t i) {

	return letter_expr<1>(*let.at(i));
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_SUBEXPR_LABEL_BUILDER_H
