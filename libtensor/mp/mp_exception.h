#ifndef LIBTENSOR_MP_EXCEPTION_H
#define LIBTENSOR_MP_EXCEPTION_H

#include "../exception.h"

namespace libtensor {


/**	\brief Exception indicating a problem with parallel processing

	\ingroup libtensor_mp
 **/
class mp_exception : public exception_base<mp_exception> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	mp_exception(const char *ns, const char *clazz,
		const char *method, const char *file, unsigned int line,
		const char *message) throw() :
		exception_base<mp_exception>(ns, clazz, method, file, line,
			"mp_exception", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~mp_exception() throw() { };

	//@}
};


} // namespace libtensor

#endif // LIBTENSOR_MP_EXCEPTION_H
