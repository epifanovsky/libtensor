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

private:
	btensor_i<N, T> &m_bt;
	letter_expr<N, Label> m_label;

public:
	/**	\brief Constructs the labeled block %tensor
	 **/
	labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N, Label> label);

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

private:
	btensor_i<N, T> &m_bt;
	letter_expr<N, Label> m_label;

public:
	/**	\brief Constructs the labeled block %tensor
	 **/
	labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N, Label> label);

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
	void operator=(const labeled_btensor_expr<N, T, Expr> &expr)
		throw(exception);

};

template<size_t N, typename T, bool Assignable, typename Label>
inline labeled_btensor<N, T, Assignable, Label>::labeled_btensor(
	btensor_i<N, T> &bt, const letter_expr<N, Label> label) :
	m_bt(bt), m_label(label) {
}

template<size_t N, typename T, bool Assignable, typename Label>
inline bool labeled_btensor<N, T, Assignable, Label>::contains(
	const letter &let) const {
	return m_label.contains(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline size_t labeled_btensor<N, T, Assignable, Label>::index_of(
	const letter &let) const throw(exception) {
	return m_label.index_of(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline const letter &labeled_btensor<N, T, Assignable, Label>::letter_at(
	size_t i) const throw(exception) {
	return m_label.letter_at(i);
}

template<size_t N, typename T, typename Label>
inline labeled_btensor<N, T, true, Label>::labeled_btensor(
	btensor_i<N, T> &bt, const letter_expr<N, Label> label) :
	m_bt(bt), m_label(label) {
}

template<size_t N, typename T, typename Label>
inline bool labeled_btensor<N, T, true, Label>::contains(
	const letter &let) const {
	return m_label.contains(let);
}

template<size_t N, typename T, typename Label>
inline size_t labeled_btensor<N, T, true, Label>::index_of(
	const letter &let) const throw(exception) {
	return m_label.index_of(let);
}

template<size_t N, typename T, typename Label>
inline const letter &labeled_btensor<N, T, true, Label>::letter_at(
	size_t i) const throw(exception) {
	return m_label.letter_at(i);
}

template<size_t N, typename T, typename Label> template<typename Expr>
void labeled_btensor<N, T, true, Label>::operator=(
	const labeled_btensor_expr<N, T, Expr> &expr) throw(exception) {
	for(size_t i = 0; i < N; i++) if(!expr.contains(letter_at(i))) {
		throw_exc("labeled_btensor<N, T, true, Label>",
			"operator=(const labeled_btensor_expr<N, T, Expr>&)",
			"Index not found in the expression");
	}
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

