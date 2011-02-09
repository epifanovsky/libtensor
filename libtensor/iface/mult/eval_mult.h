#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_MULT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_MULT_H

#include "../expr/eval_i.h"
#include "core_mult.h"
#include "mult_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates the multiplication expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam E1 LHS expression type.
	\tparam E2 RHS expression type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename E1, typename E2, bool Recip>
class eval_mult : public eval_i<N, T> {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Addition expression core type
	typedef core_mult<N, T, E1, E2, Recip> core_t;

	//!	Addition expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Evaluating container type of the first expression (A)
	typedef typename E1::eval_container_t eval_container_a_t;

	//!	Evaluating container type of the second expression (B)
	typedef typename E2::eval_container_t eval_container_b_t;

	//!	Number of %tensor arguments in expression A
	static const size_t k_narg_tensor_a =
		eval_container_a_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments in expression A
	static const size_t k_narg_oper_a =
		eval_container_a_t::template narg<oper_tag>::k_narg;

	//!	Number of %tensor arguments in expression B
	static const size_t k_narg_tensor_b =
		eval_container_b_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments in expression B
	static const size_t k_narg_oper_b =
		eval_container_b_t::template narg<oper_tag>::k_narg;

	//! Evaluating functor type
	typedef mult_eval_functor<N, T, E1, E2, Recip,
			k_narg_tensor_a, k_narg_oper_a, k_narg_tensor_b, k_narg_oper_b>
		functor_t;

	//!	Number of arguments in the expression
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	functor_t m_func; //!< Sub-expression evaluation functor

public:
	eval_mult(expression_t &expr, const letter_expr<N> &label)
		throw(exception);

	virtual ~eval_mult() { }

	//!	\name Evaluation
	//@{

	void prepare();

	void clean();

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

	arg<N, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
		throw(exception);

	//@}
};


template<size_t N, typename T, typename E1, typename E2, bool Recip>
const char *eval_mult<N, T, E1, E2, Recip>::k_clazz =
		"eval_mult<N, T, E1, E2, Recip>";

template<size_t N, typename T, typename E1, typename E2, bool Recip>
template<int Dummy>
struct eval_mult<N, T, E1, E2, Recip>::narg<oper_tag, Dummy> {
	static const size_t k_narg = 1;
};

template<size_t N, typename T, typename E1, typename E2, bool Recip>
inline eval_mult<N, T, E1, E2, Recip>::eval_mult(
	expression_t &expr, const letter_expr<N> &label)
	throw(exception) :

	m_func(expr, label) {

}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
void eval_mult<N, T, E1, E2, Recip>::prepare() {

	m_func.evaluate();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
void eval_mult<N, T, E1, E2, Recip>::clean() {

	m_func.clean();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
template<typename Tag>
arg<N, T, Tag> eval_mult<N, T, E1, E2, Recip>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";
	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}

template<size_t N, typename T, typename E1, typename E2, bool Recip>
arg<N, T, oper_tag> eval_mult<N, T, E1, E2, Recip>::get_arg(
	const oper_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const oper_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}

	return m_func.get_arg();
}



} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_MULT_H
