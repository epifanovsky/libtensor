#ifndef LIBTENSOR_ORBIT_ITERATOR_H
#define LIBTENSOR_ORBIT_ITERATOR_H

#include "defs.h"
#include "exception.h"
#include "block_iterator.h"
#include "dimensions.h"
#include "index.h"

namespace libtensor {

/**	\brief Interface to the orbit iterator event handler
	\tparam N Tensor order.

	This event hander is used by orbit_iterator<N, T> to do the actual
	work of iterating.

	\ingroup libtensor_core
 **/
template<size_t N>
class orbit_iterator_handler_i {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Virtual destructor

	 **/
	virtual ~orbit_iterator_handler_i() { }

	//@}

	//!	\name Event handling
	//@{

	/**	\brief Invoked to loads the first %index of the sequence
		\param idx Index (to be loaded).
		\param dims Dimensions.
	 **/
	virtual void on_begin(index<N> &idx,
		const dimensions<N> &dims) const = 0;

	/**	\brief Invoked to check if an %index is the last in the sequence
		\param idx Index.
		\param dims Dimensions.
	 **/
	virtual bool on_end(const index<N> &idx,
		const dimensions<N> &dims) const = 0;

	/**	\brief Invoked when the next %index from the sequence is
			required
		\param idx Index (current on input, next on output).
		\param dims Dimensions.
	 **/
	virtual void on_next(index<N> &idx,
		const dimensions<N> &dims) const = 0;

	//@}
};


/**	\brief Iterates over the orbits of a symmetry group action
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The iterator goes over each %index subset that results from the
	symmetry group action on the set of a block indexes. There are two
	states of the iterator: active (when it points at the valid index of
	a canonical block) and inactive (when in indicates that there are no
	more blocks).

	<b>Basic use</b>

	When created, the iterator points at the first element of a sequence
	of indexes. get_index() returns a reference to the current %index.
	To check whether the current %index is the last in the sequence,
	use end(). The next() method advances the iterator to the next %index.

	It is not guaranteed that get_index() will return references to
	different objects after each advancement. In fact, it is quite the
	opposite: the same object is reused. So, the user must make a personal
	copy of the %index rather than keeping the reference if it is necessary
	use the %index after advancing the iterator.

	Example

	\code
	orbit_iterator<4, double> i = ...;
	while(!i.end()) {
		index<4> idx = i.get_index();
		...
		i.next();
	}
	\endcode

	<b>Iterator event handler</b>

	The orbit iterator provides a unified mechanism of going over unique
	blocks for all types of symmetries. To achieve that, the iterator has
	to transfer the task to a symmetry-specific event handler. The handler
	implements the orbit_iterator_handler_i<N> interface.

	\ingroup libtensor
 **/
template<size_t N, typename T>
class orbit_iterator {
private:
	const orbit_iterator_handler_i<N> &m_handler; //!< Event handler
	const block_iterator_handler_i<N, T> &m_bihandler; //!< Block iterator
	index<N> m_idx; //!< Current index
	dimensions<N> m_dims; //!< Dimensions
	bool m_end; //!< Whether the end has been reached

public:
	//!	\brief Construction and destruction
	//@{

	/**	\brief Creates the iterator with given dimensions and
			event handler
		\param handler Event handler.
		\param dims Dimensions.
	 **/
	orbit_iterator(const orbit_iterator_handler_i<N> &handler,
		const block_iterator_handler_i<N, T> &bihandler,
		const dimensions<N> &dims);

	//@}

	//!	\brief Iterator
	//@{

	/**	\brief Checks whether the end of the sequence is reached
	 **/
	bool end();

	/**	\brief Advances the iterator to the next position; does nothing
			when the end is reached
	 **/
	void next();

	//@}

	//!	\name Current position
	//@{

	/**	\brief Returns the index of the current iterator position
	 **/
	const index<N> &get_index() const;

	/**	\brief Returns an iterator over blocks in the current orbit
	 */
	block_iterator<N, T> get_blocks() const;

	//@}
};


template<size_t N, typename T>
orbit_iterator<N, T>::orbit_iterator(const orbit_iterator_handler_i<N> &handler,
	const block_iterator_handler_i<N, T> &bihandler,
	const dimensions<N> &dims)
: m_handler(handler), m_bihandler(bihandler), m_dims(dims) {

	m_handler.on_begin(m_idx, m_dims);
	m_end = m_handler.on_end(m_idx, m_dims);
}


template<size_t N, typename T>
inline bool orbit_iterator<N, T>::end() {

	return m_end;
}


template<size_t N, typename T>
inline void orbit_iterator<N, T>::next() {

	if(!m_end) {
		if(!(m_end = m_handler.on_end(m_idx, m_dims)))
			m_handler.on_next(m_idx, m_dims);
	}
}


template<size_t N, typename T>
inline const index<N> &orbit_iterator<N, T>::get_index() const {

	return m_idx;
}


template<size_t N, typename T>
inline block_iterator<N, T> orbit_iterator<N, T>::get_blocks() const {

	return block_iterator<N, T>(m_bihandler, m_idx, m_dims);
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_ITERATOR_H
