#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H

#include "../../btod/btod_diag.h"
#include "../expr/anon_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the extraction of a diagonal
	\tparam N Tensor order.
	\tparam M Diagonal order.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, typename T, typename SubCore>
class diag_eval {
public:
	static const char *k_clazz; //!< Class name

	//!	Expression core type
	typedef diag_core<N, M, T, SubCore> core_t;

	//!	Expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Sub-expression core type
	typedef SubCore sub_core_t;

	//!	Sub-expression type
	typedef expr<N, T, sub_core_t> sub_expr_t;

	//!	Evaluation of the contraction
	typedef anon_eval<N, T, sub_core_t> sub_eval_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	sub_expr_t m_sub_expr; //!< Sub-expression
	sub_eval_t m_sub_eval; //!< Evaluation of the sub-expression
	permutation<N> m_perm1; //!< Permutation for argument 1
	permutation<N> m_perm2; //!< Permutation for symmetrization
	btod_diag<N, M> m_op; //!< Extraction operation
	arg<N - M + 1, T, oper_tag> m_arg; //!< Composed operation argument

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	diag_eval(expression_t &expr, const letter_expr<N> &label)
		throw(exception);

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare() throw(exception);

	template<typename Tag>
	arg<N - M + 1, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns tensor arguments
	 **/
	arg<N - M + 1, T, oper_tag> get_arg(const tensor_tag &tag,
		size_t i) const throw(exception);

private:
	static permutation<N> mk_perm(
		expression_t &expr, const letter_expr<N> &label) {

		permutation<N> perm;
		size_t i1 = label.index_of(
			expr.get_core().get_sym().letter_at(0));
		size_t i2 = label.index_of(
			expr.get_core().get_sym().letter_at(1));
		perm.permute(i1, i2);
		return perm;
	}
};


template<size_t N, size_t M, typename T, typename SubCore>
const char *diag_eval<N, M, T, SubCore>::k_clazz =
	"diag_eval<N, M, T, SubCore>";


template<size_t N, size_t M, typename T, typename SubCore>
template<int Dummy>
struct diag_eval<N, M, T, SubCore>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};


template<size_t N, size_t M, typename T, typename SubCore>
diag_eval<N, M, T, SubCore>::diag_eval(expression_t &expr,
	const letter_expr<N> &label) throw(exception) :

	m_sub_expr(expr.get_core().get_sub_expr()),
	m_sub_eval(m_sub_expr, label),
	m_perm2(mk_perm(expr, label)),
	m_arg(m_oper, 1.0) {

}


template<size_t N, size_t M, typename T, typename SubCore>
void diag_eval<N, M, T, SubCore>::prepare() throw(exception) {

	m_sub_eval.evaluate();
}


template<size_t N, size_t M, typename T, typename SubCore>
template<typename Tag>
arg<N - M + 1, T, Tag> diag_eval<N, M, T, SubCore>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";

	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, size_t M, typename T, typename SubCore>
arg<N - M + 1, T, oper_tag> diag_eval<N, M, T, SubCore>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const oper_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}

	return m_arg;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_EVAL_H
