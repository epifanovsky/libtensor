#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM22_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM22_EVAL_H

#include "../expr/anon_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the symmetrization of two pairs
		of indexes
	\tparam N Tensor order.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T, typename SubCore>
class symm22_eval {
public:
	static const char *k_clazz; //!< Class name

	//!	Expression core type
	typedef symm22_core<N, Sym, T, SubCore> core_t;

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
	permutation<N> m_perm2; //!< Permutation for symmetrized argument 2
	permutation<N> m_perm3; //!< Permutation for symmetrized argument 3
	permutation<N> m_perm4; //!< Permutation for symmetrized argument 4
	arg<N, T, tensor_tag> m_arg1; //!< Argument 1
	arg<N, T, tensor_tag> m_arg2; //!< Argument 2
	arg<N, T, tensor_tag> m_arg3; //!< Argument 3
	arg<N, T, tensor_tag> m_arg4; //!< Argument 4

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	symm22_eval(
		expression_t &expr, const letter_expr<N> &label)
		throw(exception);

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare() throw(exception);

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns tensor arguments
	 **/
	arg<N, T, tensor_tag> get_arg(const tensor_tag &tag, size_t i) const
		throw(exception);

private:
	static permutation<N> mk_perm1(
		expression_t &expr, const letter_expr<N> &label) {

		permutation<N> perm;
		size_t i1 = label.index_of(
			expr.get_core().get_sym1().letter_at(0));
		size_t i2 = label.index_of(
			expr.get_core().get_sym1().letter_at(1));
		perm.permute(i1, i2);
		return perm;
	}

	static permutation<N> mk_perm2(
		expression_t &expr, const letter_expr<N> &label) {

		permutation<N> perm;
		size_t i1 = label.index_of(
			expr.get_core().get_sym2().letter_at(0));
		size_t i2 = label.index_of(
			expr.get_core().get_sym2().letter_at(1));
		perm.permute(i1, i2);
		return perm;
	}

	static permutation<N> mk_perm3(
		expression_t &expr, const letter_expr<N> &label) {

		permutation<N> perm;
		size_t i1 = label.index_of(
			expr.get_core().get_sym1().letter_at(0));
		size_t i2 = label.index_of(
			expr.get_core().get_sym1().letter_at(1));
		size_t j1 = label.index_of(
			expr.get_core().get_sym2().letter_at(0));
		size_t j2 = label.index_of(
			expr.get_core().get_sym2().letter_at(1));
		if(!(i1 == j1 || i1 == j2 || i2 == j1 || i2 == j2))
			perm.permute(i1, i2).permute(j1, j2);
		return perm;
	}
};


template<size_t N, bool Sym, typename T, typename SubCore>
const char *symm22_eval<N, Sym, T, SubCore>::k_clazz =
	"symm22_eval<N, Sym, T, SubCore>";


template<size_t N, bool Sym, typename T, typename SubCore>
template<int Dummy>
struct symm22_eval<N, Sym, T, SubCore>::narg<tensor_tag, Dummy> {
	static const size_t k_narg = 4;
};


template<size_t N, bool Sym, typename T, typename SubCore>
symm22_eval<N, Sym, T, SubCore>::symm22_eval(
	expression_t &expr, const letter_expr<N> &label)
	throw(exception) :

	m_sub_expr(expr.get_core().get_sub_expr()),
	m_sub_eval(m_sub_expr, label),
	m_perm2(mk_perm1(expr, label)),
	m_perm3(mk_perm2(expr, label)),
	m_perm4(mk_perm3(expr, label)),
	m_arg1(m_sub_eval.get_btensor(), m_perm1, 1.0),
	m_arg2(m_sub_eval.get_btensor(), m_perm2, Sym ? 1.0 : -1.0),
	m_arg3(m_sub_eval.get_btensor(), m_perm3, Sym ? 1.0 : -1.0),
	m_arg4(m_sub_eval.get_btensor(), m_perm4,
		m_perm4.is_identity() ? 0.0 : 1.0) {

}


template<size_t N, bool Sym, typename T, typename SubCore>
void symm22_eval<N, Sym, T, SubCore>::prepare() throw(exception) {

	m_sub_eval.evaluate();
}


template<size_t N, bool Sym, typename T, typename SubCore>
template<typename Tag>
arg<N, T, Tag> symm22_eval<N, Sym, T, SubCore>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";

	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, bool Sym, typename T, typename SubCore>
arg<N, T, tensor_tag> symm22_eval<N, Sym, T, SubCore>::get_arg(
	const tensor_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const tensor_tag&, size_t)";
	switch(i) {
	case 0:
		return m_arg1;
		break;
	case 1:
		return m_arg2;
		break;
	case 2:
		return m_arg3;
		break;
	case 3:
		return m_arg4;
		break;
	default:
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM22_EVAL_H
