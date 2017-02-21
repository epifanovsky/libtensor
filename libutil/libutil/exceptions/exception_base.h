#ifndef LIBUTIL_EXCEPTION_BASE_H
#define LIBUTIL_EXCEPTION_BASE_H

#include <cstdio>
#include <cstring>
#include <typeinfo>
#include "exception.h"

namespace libutil {


/**	\brief Base exception class

	\ingroup libutil_exceptions
 **/
template<typename Exc>
class exception_base : public exception {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an %exception using full details
		\param ns Namespace name.
		\param clazz Class name.
		\param method Method name.
		\param file Source file name.
		\param line Line number in the source file.
		\param type Exception type.
		\param message Error message.
	 **/
	exception_base(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *type,
		const char *message) throw() :
		exception(ns, clazz, method, file, line, type, message) {
	}

	/**	\brief Copy constructor
	 **/
	exception_base(const exception_base<Exc> &e) throw() :
	    exception(e) {
	}

	/**	\brief Virtual destructor
	 **/
	virtual ~exception_base() throw() { };

	//@}

	/**	\brief Clones the object
	 **/
	virtual rethrowable_i *clone() const throw() {
		try {
			const Exc &e = dynamic_cast<const Exc&>(*this);
			return new Exc(e);
		} catch(...) {
			return 0;
		}
	}

	/**	\brief Throws itself
	 **/
	virtual void rethrow() const {
		const Exc &e = dynamic_cast<const Exc&>(*this);
		throw Exc(e);
	}

};


} // namespace libutil

#endif // LIBUTIL_EXCEPTION_BASE_H
