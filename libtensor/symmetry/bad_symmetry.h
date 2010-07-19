#ifndef LIBTENSOR_BAD_SYMMETRY_H
#define LIBTENSOR_BAD_SYMMETRY_H

#include "../exception.h"

namespace libtensor {


/**	\brief Exception indicating a %symmetry inconsistency

	\ingroup libtensor_symmetry
 **/
class bad_symmetry : public exception_base<bad_symmetry> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	bad_symmetry(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw() :
		exception_base<bad_symmetry>(ns, clazz, method, file, line,
			"bad_symmetry", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~bad_symmetry() throw() { };

	//@}
};


} // namespace libtensor

#endif // LIBTENSOR_BAD_SYMMETRY_H
