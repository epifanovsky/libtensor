#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_EVAL_H

#include "iface/expr/anon_eval.h"
#include "../contract/core_contract.h"
#include "../contract/eval_contract.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the contraction of two tensors
	\tparam N Order of the first %tensor (A) less contraction degree.
	\tparam M Order of the second %tensor (B) less contraction degree.
	\tparam K Number of indexes contracted.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam Expr1 First expression (A) type.
	\tparam Expr2 Second expression (B) type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
class sym_contract_eval {
public:
	static const char *k_clazz; //!< Class name
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Symmetrized contraction expression core type
	typedef sym_contract_core<N, M, K, Sym, T, E1, E2> core_t;

	//!	Symmetrized contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

	//!	Contraction expression core type
	typedef core_contract<N, M, K, T, E1, E2> contract_core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, contract_core_t> contract_expr_t;

	//!	Evaluation of the contraction
	typedef anon_eval<k_orderc, T, contract_core_t> contract_eval_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	contract_expr_t m_contract; //!< Contract expression
	contract_eval_t m_contract_eval; //!< Evaluation of the contraction
	permutation<k_orderc> m_perm; //!< Permutation for symmetrization
	btod_add<k_orderc> m_op; //!< Addition operation
	arg<k_orderc, T, oper_tag> m_arg; //!< Composed operation argument

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	sym_contract_eval(
		expression_t &expr, const letter_expr<k_orderc> &label)
		throw(exception);

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare() throw(exception);

	template<typename Tag>
	arg<N + M, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns a single argument
	 **/
	arg<N + M, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
		throw(exception);

};


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
const char *sym_contract_eval<N, M, K, Sym, T, E1, E2>::k_clazz =
	"sym_contract_eval<N, M, K, Sym, T, E1, E2>";


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
template<int Dummy>
struct sym_contract_eval<N, M, K, Sym, T, E1, E2>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
inline sym_contract_eval<N, M, K, Sym, T, E1, E2>::sym_contract_eval(
	expression_t &expr, const letter_expr<k_orderc> &label)
	throw(exception) :

	m_contract(contract_core_t(
		expr.get_core().get_contr(), expr.get_core().get_expr_1(),
		expr.get_core().get_expr_2())),
	m_contract_eval(m_contract, label),
	m_op(m_contract_eval.get_btensor()),
	m_arg(m_op, 1.0) {

	size_t i1 = label.index_of(expr.get_core().get_sym().letter_at(0));
	size_t i2 = label.index_of(expr.get_core().get_sym().letter_at(1));
	m_perm.permute(i1, i2);
	m_op.add_op(m_contract_eval.get_btensor(), m_perm, Sym ? 1.0 : -1.0);
}


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
inline void sym_contract_eval<N, M, K, Sym, T, E1, E2>::prepare()
	throw(exception) {

	m_contract_eval.evaluate();
}


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
template<typename Tag>
arg<N + M, T, Tag> sym_contract_eval<N, M, K, Sym, T, E1, E2>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";
	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, size_t M, size_t K, bool Sym, typename T,
typename E1, typename E2>
arg<N + M, T, oper_tag> sym_contract_eval<N, M, K, Sym, T, E1, E2>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const oper_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}

	return m_arg;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYM_CONTRACT_EVAL_H
