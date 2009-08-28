#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr_arg.h"
#include "letter_expr.h"

namespace libtensor {

namespace labeled_btensor_expr {

template<size_t N, typename T, bool Assignable> class eval_ident;

/**	\brief Identity expression core (references one labeled %tensor)
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Assignable Whether the %tensor is an l-value.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class core_ident {
public:
	//!	Labeled block %tensor type
	typedef labeled_btensor<N, T, Assignable> labeled_btensor_t;

	//!	Evaluating container type
	typedef eval_ident<N, T, Assignable> eval_container_t;

private:
	labeled_btensor_t &m_t; //!< Labeled block %tensor

public:
	/**	\brief Initializes the operation with a %tensor reference
	 **/
	core_ident(labeled_btensor_t &t) : m_t(t) { }

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

template<size_t N, typename T, bool Assignable>
class eval_ident {
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
template<int Dummy>
struct eval_ident<N, T, Assignable>::narg<tensor_tag, Dummy> {
	static const size_t k_narg = 1;
};

template<size_t N, typename T, bool Assignable>
inline bool core_ident<N, T, Assignable>::contains(const letter &let) const {

	return m_t.contains(let);
}

template<size_t N, typename T, bool Assignable>
inline size_t core_ident<N, T, Assignable>::index_of(const letter &let) const
	throw(exception) {

	return m_t.index_of(let);
}

template<size_t N, typename T, bool Assignable>
inline const letter&core_ident<N, T, Assignable>::letter_at(size_t i) const
	throw(exception) {

	return m_t.letter_at(i);
}

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

	throw_exc("eval_ident<N, T, Assignable>",
		"get_arg(const Tag &, size_t i)",
		"Invalid method to call.");
}

template<size_t N, typename T, bool Assignable>
inline arg<N, T, tensor_tag> eval_ident<N, T, Assignable>::get_arg(
	const tensor_tag &tag, size_t i) const throw(exception) {

	if(i != 0) {
		throw_exc("eval_ident<N, T, Assignable, Label>",
			"get_arg(size_t i)", "Invalid argument number.");
	}
	return arg<N, T, tensor_tag>(
		m_expr.get_core().get_tensor().get_btensor(), m_perm, 1.0);
}

} // namespace labeled_btensor_expr

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_H
