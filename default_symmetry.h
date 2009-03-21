#ifndef LIBTENSOR_DEFAULT_SYMMETRY_H
#define LIBTENSOR_DEFAULT_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Default symmetry implementation

	\ingroup libtensor
**/
template<size_t N>
class default_symmetry : public symmetry_i<N> {
private:
	permutation<N> m_perm; //!< Identity permutation

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Initializes default symmetry for a %tensor of
			a given order
	**/
	default_symmetry();

	/**	\brief Virtual destructor
	**/
	virtual ~default_symmetry();
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
inline default_symmetry<N>::default_symmetry() {
}

template<size_t N>
inline default_symmetry<N>::~default_symmetry() {
}

template<size_t N>
inline bool default_symmetry<N>::is_unique(const index<N> &i) const
	throw(exception) {
	return true;
}

template<size_t N>
inline const index<N> &default_symmetry<N>::get_unique(const index<N> &i) const
	throw(exception) {
	return i;
}

template<size_t N>
inline const permutation<N> &default_symmetry<N>::get_perm(const index<N> &i)
	const throw(exception) {
	return m_perm;
}

template<size_t N>
inline double default_symmetry<N>::get_coeff(const index<N> &i) const
	throw(exception) {
	return 1.0;
}

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_H

