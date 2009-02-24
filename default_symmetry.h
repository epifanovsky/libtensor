#ifndef LIBTENSOR_DEFAULT_SYMMETRY_H
#define LIBTENSOR_DEFAULT_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Default symmetry implementation

	\ingroup libtensor
**/
class default_symmetry : public symmetry_i {
private:
	permutation m_perm; //!< Identity permutation

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Initializes default symmetry for a %tensor of
			a given order
		\param order Tensor order.
	**/
	default_symmetry(const size_t order);

	/**	\brief Virtual destructor
	**/
	virtual ~default_symmetry();
	//@}

	//!	\name Implementation of symmetry_i
	//@{
	virtual bool is_unique(const index &i) const throw(exception);
	virtual const index &get_unique(const index &i) const throw(exception);
	virtual const permutation &get_perm(const index &i) const
		throw(exception);
	virtual double get_coeff(const index &i) const throw(exception);
	//@}
};

inline default_symmetry::default_symmetry(const size_t order) : m_perm(order) {
}

inline default_symmetry::~default_symmetry() {
}

inline bool default_symmetry::is_unique(const index &i) const
	throw(exception) {
	return true;
}

inline const index &default_symmetry::get_unique(const index &i) const
	throw(exception) {
	return i;
}

inline const permutation &default_symmetry::get_perm(const index &i) const
	throw(exception) {
	return m_perm;
}

inline double default_symmetry::get_coeff(const index &i) const
	throw(exception) {
	return 1.0;
}

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_H

