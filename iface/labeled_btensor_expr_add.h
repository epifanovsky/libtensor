#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_arg.h"
#include "letter_expr.h"

namespace libtensor {

namespace labeled_btensor_expr {

template<size_t N, typename T, typename CoreL, typename CoreR>
class eval_add;

/**	\brief Addition operation core expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam ExprL LHS expression type (expr).
	\tparam ExprR RHS expression type (expr).

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class core_add {
public:
	//!	Evaluating container type
	typedef eval_add<N, T, ExprL, ExprR> eval_container_t;

private:
	ExprL m_expr_l; //!< Left expression
	ExprR m_expr_r; //!< Right expression

public:
	//!	\name Construction
	//@{

	/**	\brief Initializes the core with left and right expressions
	 **/
	core_add(const ExprL &expr_l, const ExprR &expr_r)
		: m_expr_l(expr_l), m_expr_r(expr_r) { }

	//@}

	/**	\brief Returns the left expression
	 **/
	ExprL &get_expr_l() { return m_expr_l; }

	/**	\brief Returns the right expression
	 **/
	ExprR &get_expr_r() { return m_expr_r; }

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

/**	\brief Evaluates the addition expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class eval_add {
public:
	//!	Addition expression core type
	typedef core_add<N, T, ExprL, ExprR> core_t;

	//!	Addition expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Evaluating container type for the left expression
	typedef typename ExprL::eval_container_t eval_container_l_t;

	//!	Evaluating container type for the right expression
	typedef typename ExprR::eval_container_t eval_container_r_t;

	//!	Number of arguments in the expression
	template<typename Tag>
	struct narg {
		static const size_t k_narg =
			eval_container_l_t::template narg<Tag>::k_narg +
			eval_container_r_t::template narg<Tag>::k_narg;
	};

private:
	expression_t &m_expr; //!< Addition expression
	eval_container_l_t m_cont_l; //!< Left evaluating container
	eval_container_r_t m_cont_r; //!< Right evaluating container

public:
	eval_add(expression_t &expr, labeled_btensor<N, T, true> &result)
		throw(exception);

	//!	\name Evaluation
	//@{

	void prepare() throw(exception) { }

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

	//@}
};

template<size_t N, typename T, typename ExprL, typename ExprR>
inline bool core_add<N, T, ExprL, ExprR>::contains(const letter &let) const {

	return m_expr_l.contains(let);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
inline size_t core_add<N, T, ExprL, ExprR>::index_of(const letter &let) const
	throw(exception) {

	return m_expr_l.index_of(let);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
inline const letter &core_add<N, T, ExprL, ExprR>::letter_at(size_t i) const
	throw(exception) {

	return m_expr_l.letter_at(i);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
eval_add<N, T, ExprL, ExprR>::eval_add(expression_t &expr,
	labeled_btensor<N, T, true> &result) throw(exception) :
		m_expr(expr),
		m_cont_l(expr.get_core().get_expr_l(), result),
		m_cont_r(expr.get_core().get_expr_r(), result) {

}

template<size_t N, typename T, typename ExprL, typename ExprR>
template<typename Tag>
arg<N, T, Tag> eval_add<N, T, ExprL, ExprR>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	if(i > narg<Tag>::k_narg) {
		throw out_of_bounds(g_ns, "eval_add<N, T, ExprL, ExprR>",
			"get_arg(const Tag&, size_t)", __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}

	const size_t narg_l = eval_container_l_t::template narg<Tag>::k_narg;
	const size_t narg_r = eval_container_r_t::template narg<Tag>::k_narg;

	return (narg_l > 0 && narg_l > i) ?
		m_cont_l.get_arg(tag, i) : m_cont_r.get_arg(tag, i - narg_l);
}

} // namespace labeled_btensor_expr

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
