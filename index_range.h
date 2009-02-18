#ifndef __LIBTENSOR_INDEX_RANGE_H
#define __LIBTENSOR_INDEX_RANGE_H

#include "defs.h"
#include "exception.h"
#include "index.h"

namespace libtensor {

/**	\brief Defines a range of %tensor elements

	Keeps the upper-left and the lower-right indexes that define a
	range.

	\ingroup libtensor
**/
class index_range {
private:
	index m_begin; //!< Index of the first element of the range
	index m_end; //!< Index of the last element of the range

public:
	/**	\brief Creates a range using two indexes
		\param begin First %index of the range
		\param end Last %index of the range
		\throw exception if a range can't be defined by the given
			two indexes
	**/
	index_range(const index &begin, const index &end) throw(exception);

	/**	\brief Copies a range from another index_range object
	**/
	index_range(const index_range &r);

	/**	\brief Returns the first %index of the range
	**/
	const index &get_begin() const;

	/**	\brief Returns the last %index of the range
	**/
	const index &get_end() const;

	/**	\brief Checks if two %index ranges are equal

		Two ranges are equal if they have the same beginning and the
		same end.
	**/
	bool equals(const index_range &r) const;

	/**	\brief Permutes both indices defining the range
	**/
	template<class Perm>
	index_range &permute(const Perm &p) throw(exception);

private:
	/**	\brief Throws an exception
	**/
	void throw_exc(const char *method, const char *msg) throw(exception);
};

inline index_range::index_range(const index &begin, const index &end)
	throw(exception) : m_begin(begin), m_end(end) {

	register unsigned int order = begin.get_order();
#ifdef TENSOR_DEBUG
	if(order != end.get_order()) {
		throw_exc("index_range(const index&, const index&",
			"Incompatible indexes");
	}
#endif
	#pragma loop count(6)
	for(register unsigned int i=0; i<order; i++) if(m_begin[i] > m_end[i]) {
		register size_t t = m_end[i];
		m_end[i] = m_begin[i]; m_begin[i] = t;
	}
}

inline index_range::index_range(const index_range &r) :
	m_begin(r.m_begin), m_end(r.m_end) {
}

inline const index &index_range::get_begin() const {
	return m_begin;
}

inline const index &index_range::get_end() const {
	return m_end;
}

inline bool index_range::equals(const index_range &r) const {
	return (m_begin.equals(r.m_begin) && m_end.equals(r.m_end));
}

template<class Perm>
inline index_range &index_range::permute(const Perm &p)
	throw(exception) {
	m_begin.permute(p); m_end.permute(p);
	return *this;
}

inline void index_range::throw_exc(const char *method, const char *msg)
	throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::index_range::%s] %s.", method, msg);
	throw exception(s);
}

} // namespace libtensor

#endif // __LIBTENSOR_INDEX_RANGE_H

