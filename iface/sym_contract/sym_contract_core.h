#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_CORE_H

#include "defs.h"
#include "exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class sym_contract_eval;


/**	\brief Symmetrized contraction operation expression core
	\tparam N Order of the first %tensor (A) less contraction degree.
	\tparam M Order of the second %tensor (B) less contraction degree.
	\tparam K Number of indexes contracted.
	\tparam E1 First expression (A) type.
	\tparam E2 Second expression (B) type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
class sym_contract_core {
public:
	static const char *k_clazz; //!< Class name

public:
	 //!	Evaluating container type
	typedef sym_contract_eval<N, M, K, T, E1, E2> eval_container_t;

private:
	letter_expr<2> m_sym; //!< Symmetrized indexes
	letter_expr<K> m_contr; //!< Contracted indexes
	E1 m_expr1; //!< First expression
	E2 m_expr2; //!< Second expression
	const letter *m_defout[N + M]; //!< Default output label

public:
	/**	\brief Creates the expression core
		\param sym Two-letter expression indicating symmetrized indexes
		\param contr Letter expression indicating which indexes will be
			contracted.
		\param expr1 First expression (A).
		\param expr2 Second expression (B).
	 **/
	sym_contract_core(
		const letter_expr<2> &sym, const letter_expr<K> &contr,
		const E1 &expr1, const E2 &expr2);

	/**	\brief Copy constructor
	 **/
	sym_contract_core(const sym_contract_core<N, M, K, T, E1, E2> &core);

	/**	\brief Returns the symmetrized indexes
	 **/
	const letter_expr<2> &get_sym() const {
		return m_sym;
	}

	/**	\brief Returns the contracted indexes
	 **/
	const letter_expr<K> &get_contr() const {
		return m_contr;
	}

	/**	\brief Returns the first expression (A)
	 **/
	E1 &get_expr_1() {
		return m_expr1;
	}

	/**	\brief Returns the first expression (A), const version
	 **/
	const E1 &get_expr_1() const {
		return m_expr1;
	}

	/**	\brief Returns the second expression (B)
	 **/
	E2 &get_expr_2() {
		return m_expr2;
	}

	/**	\brief Returns the second expression (B), const version
	 **/
	const E2 &get_expr_2() const {
		return m_expr2;
	}

	/**	\brief Returns whether the result's label contains a %letter
		\param let Letter.
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the result's label
		\param let Letter.
		\throw expr_exception If the label does not contain the
			requested letter.
	 **/
	size_t index_of(const letter &let) const throw(expr_exception);

	/**	\brief Returns the %letter at a given position in
			the result's label
		\param i Letter index.
		\throw out_of_bounds If the index is out of bounds.
	 **/
	const letter &letter_at(size_t i) const throw(out_of_bounds);

};


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
const char *sym_contract_core<N, M, K, T, E1, E2>::k_clazz =
	"sym_contract_core<N, M, K, T, E1, E2>";


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
sym_contract_core<N, M, K, T, E1, E2>::sym_contract_core(
	const letter_expr<2> &sym, const letter_expr<K> &contr,
	const E1 &expr1, const E2 &expr2) :

	m_sym(sym), m_contr(contr), m_expr1(expr1), m_expr2(expr2) {

	static const char *method =
		"sym_contract_core(const letter_expr<2>&, "
		"const letter_expr<K>&, const E1&, const E2&)";

	if(sym.letter_at(0) == sym.letter_at(1)) {
		throw expr_exception(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Symmetrized indexes must be different.");
	}
	for(size_t i = 0; i < 2; i++) {
		const letter &l = sym.letter_at(i);
		if(contr.contains(l)) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Symmetrized index is being contracted.");
		}
		if(!expr1.contains(l) && !expr2.contains(l)) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Symmetrized index is absent from the result.");
		}
	}

	for(size_t i = 0; i < K; i++) {
		const letter &l = contr.letter_at(i);
		if(!expr1.contains(l) || !expr2.contains(l)) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Contracted index is absent from arguments.");
		}
	}

	size_t j = 0;
	for(size_t i = 0; i < N + M; i++) m_defout[i] = NULL;
	for(size_t i = 0; i < N + K; i++) {
		const letter &l = expr1.letter_at(i);
		if(!contr.contains(l)) {
			if(expr2.contains(l)) {
				throw expr_exception(g_ns, k_clazz, method,
					__FILE__, __LINE__,
					"Duplicate uncontracted index in A.");
			} else {
				m_defout[j++] = &l;
			}
		}
	}
	for(size_t i = 0; i < M + K; i++) {
		const letter &l = expr2.letter_at(i);
		if(!contr.contains(l)) {
			if(expr1.contains(l)) {
				throw expr_exception(g_ns, k_clazz, method,
					__FILE__, __LINE__,
					"Duplicate uncontracted index in B.");
			} else {
				m_defout[j++] = &l;
			}
		}
	}
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
sym_contract_core<N, M, K, T, E1, E2>::sym_contract_core(
	const sym_contract_core<N, M, K, T, E1, E2> &core) :

	m_sym(core.m_sym), m_contr(core.m_contr),
	m_expr1(core.m_expr1), m_expr2(core.m_expr2) {

	for(size_t i = 0; i < N + M; i++) {
		m_defout[i] = core.m_defout[i];
	}
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline bool sym_contract_core<N, M, K, T, E1, E2>::contains(
	const letter &let) const {

	for(register size_t i = 0; i < N + M; i++) {
		if(m_defout[i] == &let) return true;
	}
	return false;
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline size_t sym_contract_core<N, M, K, T, E1, E2>::index_of(
	const letter &let) const throw(expr_exception) {

	static const char *method = "index_of(const letter&)";

	for(register size_t i = 0; i < N + M; i++) {
		if(m_defout[i] == &let) return i;
	}

	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Expression doesn't contain the letter.");
}


template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
inline const letter &sym_contract_core<N, M, K, T, E1, E2>::letter_at(
	size_t i) const throw(out_of_bounds) {

	static const char *method = "letter_at(size_t)";

	if(i >= N + M) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Letter index is out of bounds.");
	}
	return *(m_defout[i]);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_CORE_H
