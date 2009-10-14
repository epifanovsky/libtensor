#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_SUBEXPR_LABEL_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_SUBEXPR_LABEL_BUILDER_H

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Label builder for sub-expressions in contract (base class)

	\ingroup libtensor_iface
 **/
template<size_t N, size_t M, size_t K>
class contract_subexpr_label_builder_base {
protected:
	template<typename Expr>
	static size_t next_i(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr, size_t i);

	static const letter &get_letter(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, size_t i);

};


//template<size_t N, size_t M, size_t K, size_t Cnt>
//class contract_subexpr_label_builder;
//
//template<size_t N, size_t M, size_t K>
//class contract_subexpr_label_builder<N, M, K, 1>;


/**	\brief Builds a label with which a sub-expression should be evaluated

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, size_t Cnt>
class contract_subexpr_label_builder :
	public contract_subexpr_label_builder_base<N, M, K> {
private:
	size_t m_i;
	contract_subexpr_label_builder<N, M, K, Cnt - 1> m_sub_builder;
	letter_expr<Cnt> m_label;

public:
	template<typename Expr>
	contract_subexpr_label_builder(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr);

	size_t get_i() const {
		return m_i;
	}

	const letter_expr<Cnt> &get_label() const {
		return m_label;
	}

};


template<size_t N, size_t M, size_t K>
class contract_subexpr_label_builder<N, M, K, 1> :
	public contract_subexpr_label_builder_base<N, M, K> {
private:
	size_t m_i;
	letter_expr<1> m_label;

public:
	template<typename Expr>
	contract_subexpr_label_builder(const letter_expr<N + M> &label_c,
		const letter_expr<K> &contr, const Expr &expr);

	size_t get_i() const {
		return m_i;
	}

	const letter_expr<1> &get_label() const {
		return m_label;
	}

};


template<size_t N, size_t M, size_t K> template<typename Expr>
size_t contract_subexpr_label_builder_base<N, M, K>::next_i(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr, size_t i) {

	if(i < N + M) {
		size_t j = i;
		for(; j < N + M; j++) {
			if(expr.contains(label_c.letter_at(j))) return j;
		}
		for(j = 0; j < K; j++) {
			if(expr.contains(contr.letter_at(j)))
				return j + N + M;
		}
	} else {
		size_t j = i - N - M;
		for(; j < K; j++) {
			if(expr.contains(contr.letter_at(j)))
				return j + N + M;
		}
	}
	throw_exc("contract_subexpr_label_builder_base", "next_i()",
		"Inconsistent expression");
}


template<size_t N, size_t M, size_t K>
inline const letter &contract_subexpr_label_builder_base<N, M, K>::get_letter(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	size_t i) {

	if(i < N + M) return label_c.letter_at(i);
	else return contr.letter_at(i - N - M);
}


template<size_t N, size_t M, size_t K, size_t Cur> template<typename Expr>
contract_subexpr_label_builder<N, M, K, Cur>::contract_subexpr_label_builder(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr) :
	m_sub_builder(label_c, contr, expr),
	m_i(next_i(label_c, contr, expr, m_sub_builder.get_i() + 1)),
	m_label(m_sub_builder.get_label(), get_letter(label_c, contr, m_i)) {

}


template<size_t N, size_t M, size_t K> template<typename Expr>
contract_subexpr_label_builder<N, M, K, 1>::contract_subexpr_label_builder(
	const letter_expr<N + M> &label_c, const letter_expr<K> &contr,
	const Expr &expr) :
	m_i(next_i(label_c, contr, expr, 0)),
	m_label(get_letter(label_c, contr, m_i)) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_SUBEXPR_LABEL_BUILDER_H
