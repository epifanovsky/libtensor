#ifndef LIBTENSOR_LABELED_BTENSOR_H
#define	LIBTENSOR_LABELED_BTENSOR_H

#include "defs.h"
#include "exception.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T, typename Traits> class btensor;

/**	\brief Block %tensor with an attached label
	\tparam N Tensor order.
	\tparam Traits Tensor traits.
	\tparam LabelT Label expression.

	\ingroup libtensor
 **/
template<size_t N, typename T, typename Traits, typename LabelT>
class labeled_btensor {
private:
	typedef T element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	btensor<N, T, Traits> &m_t;
	letter_expr<N, LabelT> m_label;

public:
	labeled_btensor(btensor<N, T, Traits> &t,
		const letter_expr<N, LabelT> label);

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

template<size_t N, typename T, typename Traits, typename LabelT>
inline labeled_btensor<N, T, Traits, LabelT>::labeled_btensor(
	btensor<N, T, Traits> &t, const letter_expr<N, LabelT> label) :
	m_t(t), m_label(label) {
}

template<size_t N, typename T, typename Traits, typename LabelT>
inline bool labeled_btensor<N, T, Traits, LabelT>::contains(
	const letter &let) const {
	return m_label.contains(let);
}

template<size_t N, typename T, typename Traits, typename LabelT>
inline size_t labeled_btensor<N, T, Traits, LabelT>::index_of(
	const letter &let) const throw(exception) {
	return m_label.index_of(let);
}

template<size_t N, typename T, typename Traits, typename LabelT>
inline const letter &labeled_btensor<N, T, Traits, LabelT>::letter_at(
	size_t i) const throw(exception) {
	return m_label.letter_at(i);
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

