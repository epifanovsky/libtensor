#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr_arg.h"

namespace libtensor {

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

public:
	//!	\brief Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = 1;

	//!	\brief Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = 0;

private:
	labeled_btensor_t &m_t;

public:

	//!	\name Construction
	//@{

	/**	\brief Initializes the operation with a %tensor reference
	 **/
	labeled_btensor_expr_ident(labeled_btensor_t &t);

	//@}

	//!	\name Evaluation
	//@{

	/**	\brief Returns the %tensor argument
		\tparam Label Label expression (to figure out the %permutation)
		\param i Argument number (0 is the only allowed value)
	 **/
	template<typename Label2>
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i) const
		throw(exception);

	//@}
};

template<size_t N, typename T, bool Assignable, typename Label>
inline labeled_btensor_expr_ident<N, T, Assignable, Label>::
labeled_btensor_expr_ident(labeled_btensor_t &t)
	: m_t(t) {
}

template<size_t N, typename T, bool Assignable, typename Label>
template<typename Label2>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_expr_ident<N, T, Assignable, Label>::get_arg_tensor(size_t i)
	const throw(exception) {
	permutation<N> perm;
	if(i == 0) {
		return labeled_btensor_expr_arg_tensor<N, T>(
			m_t.get_btensor(), perm, 1.0);
	} else {
		throw_exc("labeled_btensor_expr_ident<N, T, Assignable, Label>",
			"get_arg(size_t i)", "Invalid argument number");
	}
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
