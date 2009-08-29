#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_IDENT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_IDENT_H

#include "core_ident.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluating container for a labeled %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Assignable Whether the %tensor is an l-value.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class eval_ident {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Expression core type
	typedef core_ident<N, T, Assignable> core_t;

	//!	Expression type
	typedef expr<N, T, core_t> expression_t;

	//!	Number of arguments
	template<typename Tag, int Dummy = 0>
	struct narg {
		static const size_t k_narg = 0;
	};

private:
	expression_t &m_expr;
	permutation<N> m_perm;

public:
	eval_ident(expression_t &expr, labeled_btensor<N, T, true> &result)
		throw(exception);

	//!	\name Evaluation
	//@{

	void prepare() throw(exception) { }

	template<typename Tag>
	arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

	/**	\brief Returns the %tensor argument
		\param i Argument number (0 is the only allowed value)
	 **/
	arg<N, T, tensor_tag> get_arg(const tensor_tag &tag, size_t i) const
		throw(exception);

	//@}
};


template<size_t N, typename T, bool Assignable>
const char *eval_ident<N, T, Assignable>::k_clazz =
	"eval_ident<N, T, Assignable>";


template<size_t N, typename T, bool Assignable>
template<int Dummy>
struct eval_ident<N, T, Assignable>::narg<tensor_tag, Dummy> {
	static const size_t k_narg = 1;
};


template<size_t N, typename T, bool Assignable>
eval_ident<N, T, Assignable>::eval_ident(expression_t &expr,
	labeled_btensor<N, T, true> &result) throw(exception) :
		m_expr(expr),
		m_perm(result.get_label().permutation_of(
			expr.get_core().get_tensor().get_label())) {

}


template<size_t N, typename T, bool Assignable>
template<typename Tag>
inline arg<N, T, Tag> eval_ident<N, T, Assignable>::get_arg(
	const Tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const Tag&, size_t)";
	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Invalid method.");
}


template<size_t N, typename T, bool Assignable>
inline arg<N, T, tensor_tag> eval_ident<N, T, Assignable>::get_arg(
	const tensor_tag &tag, size_t i) const throw(exception) {

	static const char *method = "get_arg(const tensor_tag&, size_t)";

	if(i != 0) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Argument index is out of bounds.");
	}
	return arg<N, T, tensor_tag>(
		m_expr.get_core().get_tensor().get_btensor(), m_perm, 1.0);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_IDENT_H
