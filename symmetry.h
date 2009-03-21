#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <map>

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "lehmer_code.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Stores %symmetry information about blocks in a block %tensor

	The concept of block %symmetry described in libtensor::symmetry_i.

	\ingroup libtensor
**/
template<size_t N>
class symmetry : public symmetry_i<N> {
private:
	struct syminfo {
		size_t unique;
		size_t perm;
		double coeff;
	};
	typedef typename std::map<size_t,syminfo> symmap;

	dimensions<N> m_dims; //!< Dimensions of blocks
	symmap m_sym; //!< Stores replicas

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Makes a deep copy of another symmetry object
		\param s Symmetry.
		\param d Dimensions to which the symmetry is applied.
	**/
	symmetry(const symmetry_i<N> &s, const dimensions<N> &d);

	/**	\brief Virtual destructor
	**/
	virtual ~symmetry();

	//@}

	//!	\name Implementation of symmetry_i
	//@{
	virtual bool is_unique(const index<N> &i) const throw(exception);
	virtual const index<N> &get_unique(const index<N> &i) const
		throw(exception);
	virtual const permutation<N> &get_perm(const index<N> &i) const
		throw(exception);
	virtual double get_coeff(const index<N> &i) const throw(exception);
	//@}
};

template<size_t N>
symmetry<N>::symmetry(const symmetry_i<N> &s, const dimensions<N> &d) :
	m_dims(d) {
	syminfo si;
	index<N> i;
	do {
		if(!s.is_unique(i)) {
			size_t iabs = d.abs_index(i);
			si.unique = d.abs_index(s.get_unique(i));
			si.perm = lehmer_code<N>::get_instance().
				perm2code(s.get_perm(i));
			si.coeff = s.get_coeff(i);
			m_sym[iabs] = si;
		}
	} while(d.inc_index(i));
}

template<size_t N>
symmetry<N>::~symmetry() {
}

template<size_t N>
bool symmetry<N>::is_unique(const index<N> &i) const throw(exception) {
	return m_sym.find(m_dims.abs_index(i)) == m_sym.end();
}

/**	\todo Implement symmetry::get_unique
**/
template<size_t N>
const index<N> &symmetry<N>::get_unique(const index<N> &i) const
	throw(exception) {
	typename std::map<size_t,syminfo>::const_iterator iter =
		m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) return i;

	return i;
}

template<size_t N>
const permutation<N> &symmetry<N>::get_perm(const index<N> &i) const
	throw(exception) {
	typename std::map<size_t,syminfo>::const_iterator iter =
		m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) {
		return lehmer_code<N>::get_instance().code2perm(0);
	} else {
		return lehmer_code<N>::get_instance().code2perm(
			iter->second.perm);
	}
}

template<size_t N>
double symmetry<N>::get_coeff(const index<N> &i) const throw(exception) {
	typename std::map<size_t,syminfo>::const_iterator iter =
		m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) return 1.0;
	else return iter->second.coeff;
}

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H

