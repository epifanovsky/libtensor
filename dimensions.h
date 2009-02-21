#ifndef LIBTENSOR_DIMENSIONS_H
#define LIBTENSOR_DIMENSIONS_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "index_range.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Contains %tensor %dimensions

	Stores the number of %tensor elements along each dimension.

	\ingroup libtensor
**/
class dimensions {
private:
	index m_dims; //!< Tensor %dimensions
	index m_incs; //!< Index increments
	size_t m_size; //!< Total size

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates a copy of another dimensions object
		\param d Another dimensions object.
	**/
	dimensions(const dimensions &d);

	/**	\brief Convers a range of indexes to the dimensions object
		\param r Index range
	**/
	dimensions(const index_range &r);

	//@}

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
	dimensions &permute(const permutation &p) throw(exception);

	/**	\brief Increments an %index within the %dimensions
		\param i Index.
		\return True on success and false if the index cannot be
			incremented (points to the last element or out of
			bounds).
		\throw exception If the index is incompatible with the
			dimensions object.
	**/
	bool inc_index(index &i) const throw(exception);

private:
	/**	\brief Updates the linear increments for each dimension
	**/
	void update_increments();

	/**     \brief Throws an exception with an error message
	**/
	void throw_exc(const char *method, const char *msg) const
		throw(exception);
};

inline dimensions::dimensions(const dimensions &d) :
	m_dims(d.m_dims), m_incs(d.m_incs), m_size(d.m_size) {
}

inline dimensions::dimensions(const index_range &r) :
	m_dims(r.get_end()), m_incs(m_dims.get_order()) {
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_dims.get_order(); i++) {
		m_dims[i] -= r.get_begin()[i];
		m_dims[i] ++;
	}
	update_increments();
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

inline dimensions &dimensions::permute(const permutation &p) throw(exception) {
	m_dims.permute(p);
	update_increments();
	return *this;
}

inline bool dimensions::inc_index(index &i) const throw(exception) {
	if(m_dims.get_order() != i.get_order())
		throw_exc("inc_index(index&)", "Incompatible index");
	if(m_dims.less(i) || m_dims.equals(i)) return false;
	size_t n = m_dims.get_order() - 1;
	bool done = false;
	while(!done && n!=0) {
		if(i[n] < m_dims[n]-1) {
			i[n]++;
			for(size_t j=n+1; j<m_dims.get_order(); j++) i[j]=0;
			done = true;
		} else {
			n--;
		}
	}
	return done;
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

inline void dimensions::throw_exc(const char *method, const char *msg) const
	throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::dimensions::%s] %s.", method, msg);
	throw exception(s);
}

} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_H

