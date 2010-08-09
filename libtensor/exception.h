#ifndef LIBTENSOR_EXCEPTION_H
#define LIBTENSOR_EXCEPTION_H

#include <exception>
#include "backtrace.h"

namespace libtensor {

/**	\defgroup libtensor_core_exc Exceptions
	\ingroup libtensor_core
 **/

/**	\brief Base %exception class

	This is the base class of a hierarchy of tensor library exceptions.
	The descendants of the hierarchy are specific exceptions that indicate
	the cause. The base class provides constructors and methods that compose
	an error message using details such as the class and method where
	the exception occurred, and a line that describes the error.
	The message is available through the what() method of the std::exception
	class, consists of two lines, and has the form:

	\code
	namespace::class::method(arg, arg), source_file.C (123), exception_type
	Error message.
	\endcode

	\ingroup libtensor_core_exc
 **/
class exception : public std::exception {
private:
	char m_ns[128]; //!< Namespace name
	char m_clazz[128]; //!< Class name
	char m_method[128]; //!< Method name
	char m_file[128]; //!< Source file name
	unsigned int m_line; //!< Line number
	char m_type[128]; //!< Exception type
	char m_message[256]; //!< Exception message
	char m_what[1024]; //!< Composed message available via what()
	backtrace m_trace; //!< Stack backtrace

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
	exception(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *type,
		const char *message) throw();

	/**	\brief Virtual destructor
	 **/
	virtual ~exception() throw() { };

	//@}


	//!	\name Implementation of std::exception
	//@{

	/**	\brief Returns the cause of the exception (message)
	 **/
	virtual const char *what() const throw();

	//@}

	/**	\brief Clones the exception
	 **/
	virtual exception *clone() const throw() = 0;

	/**	\brief Throws itself
	 **/
	virtual void rethrow() = 0;

	const backtrace &get_backtrace() const {
		return m_trace;
	}

};


template<typename T>
class exception_base : public exception {
public:
	exception_base(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *type,
		const char *message) throw() : exception(ns, clazz, method,
		file, line, type, message) { }

	virtual ~exception_base() throw() { }

	virtual exception *clone() const throw() {
		return new T(dynamic_cast<const T&>(*this));
	}

	virtual void rethrow() {
		throw T(dynamic_cast<const T&>(*this));
	}

};


/**	\brief Generic exception class

	\ingroup libtensor_core_exc
 **/
class generic_exception : public exception_base<generic_exception> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	generic_exception(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<generic_exception>(ns, clazz, method,
			file, line, "generic_exception", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~generic_exception() throw() { };

	//@}

};


/**	\brief Exception indicating an invalid argument or input parameter

	\ingroup libtensor_core_exc
 **/
class bad_parameter : public exception_base<bad_parameter> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	bad_parameter(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<bad_parameter>(ns, clazz, method, file, line,
			"bad_parameter", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~bad_parameter() throw() { };

	//@}

};


/**	\brief Exception indicating that a block of a %tensor does not exist

	\ingroup libtensor_core_exc
 **/
class block_not_found : public exception_base<block_not_found> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	block_not_found(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<block_not_found>(ns, clazz, method, file, line,
			"block_not_found", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~block_not_found() throw() { };

	//@}

};


/**	\brief Exception indicating that access to an immutable object is
		violated

	\ingroup libtensor_core_exc
 **/
class immut_violation : public exception_base<immut_violation> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	immut_violation(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<immut_violation>(ns, clazz, method, file, line,
			"immut_violation", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~immut_violation() throw() { };

	//@}
};


/**	\brief Exception indicating that not enough memory is available to
		continue

	\ingroup libtensor_core_exc
 **/
class out_of_memory : public exception_base<out_of_memory> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	out_of_memory(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception_base<out_of_memory>(ns, clazz, method, file, line,
			"out_of_memory", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~out_of_memory() throw() { };

	//@}
};


/**	\brief Exception indicating that a request violates symmetry

	\ingroup libtensor_core_exc
 **/
class symmetry_violation : public exception_base<symmetry_violation> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	symmetry_violation(const char *ns, const char *clazz,
		const char *method, const char *file, unsigned int line,
		const char *message) throw()
		: exception_base<symmetry_violation>(ns, clazz, method,
			file, line, "symmetry_violation", message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_violation() throw() { };

	//@}
};


/**	\brief Throws an exception with a given error message
 **/
void throw_exc(const char *clazz, const char *method, const char *error)
	throw(exception);

} // namespace libtensor

#endif // LIBTENSOR_EXCEPTION_H

