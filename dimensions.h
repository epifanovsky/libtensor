#ifndef __LIBTENSOR_DIMENSIONS_H
#define __LIBTENSOR_DIMENSIONS_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "index_range.h"

namespace libtensor {

/**	\brief Contains %tensor %dimensions

	Keeps the %index of the last element of a %tensor. Also keeps track
	of linear increments along each dimension of the %tensor.

	\ingroup libtensor
**/
class dimensions {
private:
	index m_dims; //!< Tensor %dimensions
	index m_incs; //!< Index increments
	size_t m_size; //!< Total size

public:
	/**	\brief Creates a copy of another dimensions object
		\param d Another dimensions object.
	**/
	dimensions(const dimensions &d);

	/**	\brief Convers a range of indexes to the dimensions object
		\param r Index range
	**/
	dimensions(const index_range &r);

	/**	\brief Returns the number of dimensions
	**/
	size_t get_order() const;

	/**	\brief Returns the linear increment along a given dimension
	**/
	size_t get_increment(const size_t i) const throw(exception);

	/**	\brief Returns the number of elements along a given dimension
	**/
	size_t operator[](const size_t i) const throw(exception);

	/**	\brief Returns the total number of elements
	**/
	size_t get_size() const;

	/**	\brief Permutes the dimensions
	**/
	template<class Perm>
	dimensions &permute(const Perm &p) throw(exception);

private:
	/**	\brief Updates the linear increments for each dimension
	**/
	void update_increments();
};

inline dimensions::dimensions(const dimensions &d) :
	m_dims(d.m_dims), m_incs(d.m_incs), m_size(d.m_size) {
}

inline dimensions::dimensions(const index_range &r) :
	m_dims(r.get_end()), m_incs(m_dims.get_order()) {
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_dims.get_order(); i++)
		m_dims[i] -= r.get_begin()[i];
}

inline size_t dimensions::get_increment(const size_t i) const
	throw(exception) {
	return m_incs[i];
}

inline size_t dimensions::operator[](const size_t i) const
	throw(exception) {
	return m_dims[i];
}

inline unsigned int dimensions::get_order() const {
	return m_dims.get_order();
}

inline size_t dimensions::get_size() const {
	return m_size;
}

template<class Perm>
inline dimensions &dimensions::permute(const Perm &p) throw(exception) {
	m_dims.permute(p);
	update_increments();
}

inline void dimensions::update_increments() {
	register size_t sz = 1;
	register size_t i = m_dims.get_order();
	do {
		i--;
		m_incs[i] = sz; sz *= m_dims[i];
	} while(i != 0);
	m_size = sz;
}

} // namespace libtensor

#endif // __LIBTENSOR_DIMENSIONS_H

