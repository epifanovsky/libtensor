#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_EVAL_H

#include "../expr/eval_i.h"
#include "../expr/anon_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for the symmetrization of a pair of indexes
	\tparam N Tensor order.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T, typename SubCore>
class symm2_eval : public eval_i<N, T> {
public:
	static const char *k_clazz; //!< Class name

	//!	Expression core type
	typedef symm2_core<N, Sym, T, SubCore> core_t;

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
	arg<N, T, tensor_tag> m_arg1; //!< Argument 1
	arg<N, T, tensor_tag> m_arg2; //!< Argument 2

public:
	/**	\brief Initializes the container with given expression and
			result recipient
	 **/
	symm2_eval(
		expression_t &expr, const letter_expr<N> &label)
		throw(exception);

	/**	\brief Virtual destructor
	 **/
	virtual ~symm2_eval() { }

	/**	\brief Evaluates sub-expressions into temporary tensors
	 **/
	void prepare();

	/**	\brief Cleans up temporary tensors
	 **/
	void clean();

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const
		throw(exception);

	/**	\brief Returns tensor arguments
	 **/
	arg<N, T, tensor_tag> get_arg(const tensor_tag &tag, size_t i) const
		throw(exception);

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


template<size_t N, bool Sym, typename T, typename SubCore>
const char *symm2_eval<N, Sym, T, SubCore>::k_clazz =
	"symm2_eval<N, Sym, T, SubCore>";


template<size_t N, bool Sym, typename T, typename SubCore>
template<int Dummy>
struct symm2_eval<N, Sym, T, SubCore>::narg<tensor_tag, Dummy> {
	static const size_t k_narg = 2;
};


template<size_t N, bool Sym, typename T, typename SubCore>
symm2_eval<N, Sym, T, SubCore>::symm2_eval(
	expression_t &expr, const letter_expr<N> &label)
	throw(exception) :

	m_sub_expr(expr.get_core().get_sub_expr()),
	m_sub_eval(m_sub_expr, label),
	m_perm2(mk_perm(expr, label)),
	m_arg1(m_sub_eval.get_btensor(), m_perm1, 1.0),
	m_arg2(m_sub_eval.get_btensor(), m_perm2, Sym ? 1.0 : -1.0) {

}


template<size_t N, bool Sym, typename T, typename SubCore>
void symm2_eval<N, Sym, T, SubCore>::prepare() {

	m_sub_eval.evaluate();
}


template<size_t N, bool Sym, typename T, typename SubCore>
void symm2_eval<N, Sym, T, SubCore>::clean() {

	m_sub_eval.clean();
}


template<size_t N, bool Sym, typename T, typename SubCore>
template<typename Tag>
arg<N, T, Tag> symm2_eval<N, Sym, T, SubCore>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";

	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, bool Sym, typename T, typename SubCore>
arg<N, T, tensor_tag> symm2_eval<N, Sym, T, SubCore>::get_arg(
	const tensor_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const tensor_tag&, size_t)";
	if(i == 0) {
		return m_arg1;
	} else if(i == 1) {
		return m_arg2;
	} else {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_EVAL_H
