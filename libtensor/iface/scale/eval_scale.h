#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_SCALE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_SCALE_H

#include "../expr/eval_i.h"
#include "core_scale.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates a scaled expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Expr Underlying expression type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr>
class eval_scale : public eval_i<N, T> {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Scaling expression core type
	typedef core_scale<N, T, Expr> core_t;

	//!	Scaled expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Unscaled expression evaluating container type
	typedef typename Expr::eval_container_t unscaled_eval_container_t;

	//!	Number of arguments in the expression
	template<typename Tag>
	struct narg {
		static const size_t k_narg =
			unscaled_eval_container_t::template narg<Tag>::k_narg;
	};

private:
	expression_t &m_expr; //!< Scaled expression

	//!	Unscaled expression evaluating container
	unscaled_eval_container_t m_unscaled_cont;

public:
	//!	\name Construction
	//@{

	/**	\brief Constructs the evaluating container
	 **/
	eval_scale(expression_t &expr, const letter_expr<N> &label)
		throw(exception);

	/**	\brief Virtual destructor
	 **/
	virtual ~eval_scale() { }

	//@}

	//!	\name Evaluation
	//@{

	void prepare();

	void clean();

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

	//@}
};


template<size_t N, typename T, typename Expr>
const char *eval_scale<N, T, Expr>::k_clazz = "eval_scale<N, T, Expr>";


template<size_t N, typename T, typename Expr>
eval_scale<N, T, Expr>::eval_scale(
	expression_t &expr, const letter_expr<N> &label) throw(exception) :
		m_expr(expr),
		m_unscaled_cont(expr.get_core().get_unscaled_expr(), label) {

}


template<size_t N, typename T, typename Expr>
void eval_scale<N, T, Expr>::prepare() {

	m_unscaled_cont.prepare();
}


template<size_t N, typename T, typename Expr>
void eval_scale<N, T, Expr>::clean() {

	m_unscaled_cont.clean();
}


template<size_t N, typename T, typename Expr>
template<typename Tag>
inline arg<N, T, Tag> eval_scale<N, T, Expr>::get_arg(const Tag &tag, size_t i)
	const throw(exception) {

	arg<N, T, Tag> argument = m_unscaled_cont.get_arg(tag, i);
	argument.scale(m_expr.get_core().get_coeff());
	return argument;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_SCALE_H
