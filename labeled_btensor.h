#ifndef LIBTENSOR_LABELED_BTENSOR_H
#define	LIBTENSOR_LABELED_BTENSOR_H

#include "defs.h"
#include "exception.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T> class btensor_i;
template<size_t N, typename T, typename Expr> class labeled_btensor_expr;

/**	\brief Block %tensor with an attached label
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Assignable Whether the %tensor can be an l-value.
	\tparam Label Label expression.

	\ingroup libtensor
 **/
template<size_t N, typename T, bool Assignable, typename Label>
class labeled_btensor {
private:
	typedef T element_t;
	typedef letter_expr<N, Label> label_t;

private:
	btensor_i<N, T> &m_bt;
	letter_expr<N, Label> m_label;

public:
	/**	\brief Constructs the labeled block %tensor
	 **/
	labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N, Label> label);

	/**	\brief Returns the tensor interface
	 **/
	btensor_i<N, T> &get_btensor() const;

	/**	\brief Returns the label
	 **/
	const letter_expr<N, Label> &get_label() const;

	/**	\brief Returns whether the label contains a %letter
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the label
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position in the label
	 **/
	const letter &letter_at(size_t i) const throw(exception);

};

/**	\brief Partial specialization of the assignable labeled tensor

	\ingroup libtensor
 **/
template<size_t N, typename T, typename Label>
class labeled_btensor<N, T, true, Label> {
private:
	typedef T element_t;
	typedef letter_expr<N, Label> label_t;

private:
	btensor_i<N, T> &m_bt;
	letter_expr<N, Label> m_label;

public:
	/**	\brief Constructs the labeled block %tensor
	 **/
	labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N, Label> label);

	/**	\brief Returns the tensor interface
	 **/
	btensor_i<N, T> &get_btensor() const;

	/**	\brief Returns the label
	 **/
	const letter_expr<N, Label> &get_label() const;

	/**	\brief Returns whether the label contains a %letter
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the label
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position in the label
	 **/
	const letter &letter_at(size_t i) const throw(exception);

	/**	\brief Assigns this %tensor to an expression
	 **/
	template<typename Expr>
	labeled_btensor<N, T, true, Label> operator=(
		const labeled_btensor_expr<N, T, Expr> &rhs) throw(exception);

	template<bool AssignableR, typename LabelR>
	labeled_btensor<N, T, true, Label> operator=(
		labeled_btensor<N, T, AssignableR, LabelR> rhs)
		throw(exception);

	labeled_btensor<N, T, true, Label> operator=(
		labeled_btensor<N, T, true, Label> rhs)
		throw(exception);

};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

