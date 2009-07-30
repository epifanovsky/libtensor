#ifndef LIBTENSOR_BLOCK_ITERATOR_H
#define LIBTENSOR_BLOCK_ITERATOR_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "index.h"

namespace libtensor {

/**	\brief Describes how the the canonical block needs to be modified to
		 obtain a replica
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This template is a structure placeholder. It is to be specialized for
	each %tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
struct block_symop {

};


template<size_t N, typename T>
class block_iterator_handler_i {
public:
	virtual ~block_iterator_handler_i() { };
	virtual bool on_begin(index<N> &idx, block_symop<N, T> &symop,
		const index<N> &orbit) const = 0;
	virtual bool on_next(index<N> &idx, block_symop<N, T> &symop,
		const index<N> &orbit) const = 0;
};


template<size_t N, typename T>
class block_iterator {
private:
	const block_iterator_handler_i<N, T> &m_handler;
	index<N> m_idx;
	index<N> m_orbit;
	block_symop<N, T> m_symop;
	bool m_end;

public:
	block_iterator(const block_iterator_handler_i<N, T> &handler,
		const index<N> &orbit);

	bool end();
	void next();
	const index<N> &get_index() const;
	const block_symop<N, T> &get_symop() const;
};


template<size_t N, typename T>
block_iterator<N, T>::block_iterator(
	const block_iterator_handler_i<N, T> &handler, const index<N> &orbit)
: m_handler(handler), m_orbit(orbit) {

	m_end = !m_handler.on_begin(m_idx, m_symop, m_orbit);
}


template<size_t N, typename T>
inline bool block_iterator<N, T>::end() {

	return m_end;
}


template<size_t N, typename T>
inline void block_iterator<N, T>::next() {

	if(!m_end) {
		m_end = !m_handler.on_next(m_idx, m_symop, m_orbit);
	}
}


template<size_t N, typename T>
inline const index<N> &block_iterator<N, T>::get_index() const {

	return m_idx;
}


template<size_t N, typename T>
inline const block_symop<N, T> &block_iterator<N, T>::get_symop() const {

	return m_symop;
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_ITERATOR_H
