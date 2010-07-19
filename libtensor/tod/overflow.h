#ifndef LIBTENSOR_OVERFLOW_H
#define LIBTENSOR_OVERFLOW_H

#include "../exception.h"

namespace libtensor {

/**	\brief Exception indicating a buffer overflow

	\ingroup libtensor_core_exc
 **/
class overflow : public exception_base<overflow> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	overflow(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<overflow>(ns, clazz, method, file, line,
			"overflow", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~overflow() throw() { };

	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_OVERFLOW_H
