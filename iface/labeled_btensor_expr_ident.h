#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr_arg.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T, bool Assignable, typename Label>
class labeled_btensor_eval_ident;

/**	\brief Identity expression core (references one labeled %tensor)
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Assignable Whether the %tensor is an l-value.
	\tparam Label Label expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable, typename Label>
class labeled_btensor_expr_ident {
public:
	//!	Labeled block %tensor type
	typedef labeled_btensor<N, T, Assignable, Label> labeled_btensor_t;

	//!	Evaluating container type
	typedef labeled_btensor_eval_ident<N, T, Assignable, Label>
		eval_container_t;

private:
	labeled_btensor_t &m_t; //!< Labeled block %tensor

public:
	/**	\brief Initializes the operation with a %tensor reference
	 **/
	labeled_btensor_expr_ident(labeled_btensor_t &t) : m_t(t) { }

	/**	\brief Returns the labeled block %tensor
	 **/
	labeled_btensor_t &get_tensor() { return m_t; }

	/**	\brief Returns whether the %tensor's label contains a %letter
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the %tensor's label
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position in
			the %tensor's label
	 **/
	const letter &letter_at(size_t i) const throw(exception);

};

template<size_t N, typename T, bool Assignable, typename Label>
class labeled_btensor_eval_ident {
public:
	//!	Expression core type
	typedef labeled_btensor_expr_ident<N, T, Assignable, Label> core_t;

	//!	Expression type
	typedef labeled_btensor_expr<N, T, core_t> expression_t;

	//!	Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = 1;

	//!	Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = 0;

private:
	expression_t &m_expr;
	permutation<N> m_perm;

public:
	template<typename LabelLhs>
	labeled_btensor_eval_ident(expression_t &expr,
		labeled_btensor<N, T, true, LabelLhs> &result) throw(exception);

	//!	\name Evaluation
	//@{

	/**	\brief Returns the %tensor argument
		\param i Argument number (0 is the only allowed value)
	 **/
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i) const
		throw(exception);

	/**	\brief Returns the %tensor argument, simply causes an exception
	 **/
	labeled_btensor_expr_arg_oper<N, T> get_arg_oper(size_t i) const
		throw(exception);

	//@}
};

template<size_t N, typename T, bool Assignable, typename Label>
inline bool labeled_btensor_expr_ident<N, T, Assignable, Label>::contains(
	const letter &let) const {

	return m_t.contains(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline size_t labeled_btensor_expr_ident<N, T, Assignable, Label>::index_of(
	const letter &let) const throw(exception) {

	return m_t.index_of(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline const letter&
labeled_btensor_expr_ident<N, T, Assignable, Label>::letter_at(
	size_t i) const throw(exception) {

	return m_t.letter_at(i);
}

template<size_t N, typename T, bool Assignable, typename Label>
template<typename LabelLhs>
labeled_btensor_eval_ident<N, T, Assignable, Label>::labeled_btensor_eval_ident(
	expression_t &expr, labeled_btensor<N, T, true, LabelLhs> &result)
	throw(exception)
	: m_expr(expr), m_perm(result.get_label().permutation_of(expr.get_core().get_tensor().get_label())) {

}

template<size_t N, typename T, bool Assignable, typename Label>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_eval_ident<N, T, Assignable, Label>::get_arg_tensor(size_t i)
	const throw(exception) {

	if(i != 0) {
		throw_exc("labeled_btensor_eval_ident<N, T, Assignable, Label>",
			"get_arg_tensor(size_t i)", "Invalid argument number");
	}
	return labeled_btensor_expr_arg_tensor<N, T>(
			m_expr.get_core().get_tensor().get_btensor(), m_perm, 1.0);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline labeled_btensor_expr_arg_oper<N, T>
labeled_btensor_eval_ident<N, T, Assignable, Label>::get_arg_oper(size_t i)
	const throw(exception) {

	throw_exc("labeled_btensor_eval_ident<N, T, Assignable, Label>",
		"get_arg_oper(size_t i)", "Invalid method to call");
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
