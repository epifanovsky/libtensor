#ifndef LIBTENSOR_CUDA_EXCEPTION_H
#define LIBTENSOR_CUDA_EXCEPTION_H

#include "../exception.h"

namespace libtensor {


/**	\brief Inconsistency detected by the cuda allocator

	\ingroup libtensor
 **/
class cuda_exception : public exception_base<cuda_exception> {
private:
	typedef exception_base<cuda_exception> exception_base_t;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	cuda_exception(const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw() :
		exception_base_t("libtensor", clazz, method, file, line,
			"cuda_exception", message) { };

	/**	\brief Copy constructor
	 **/
	cuda_exception(const cuda_exception &e) throw() : exception_base_t(e) { }

	/**	\brief Virtual destructor
	 **/
	virtual ~cuda_exception() throw() { };

	//@}

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_EXCEPTION_H
