#ifndef LIBTENSOR_EXCEPTION_H
#define LIBTENSOR_EXCEPTION_H

#include <exception>

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
	char m_what[1024]; //! Composed message available via what()

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

};


/**	\brief Exception indicating an invalid argument or input parameter

	\ingroup libtensor_core_exc
 **/
class bad_parameter : public exception {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	bad_parameter(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception(ns, clazz, method, file, line, "bad_parameter",
			message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~bad_parameter() throw() { };

	//@}
};


/**	\brief Exception indicating that an %index is out of bounds

	\ingroup libtensor_core_exc
 **/
class out_of_bounds : public exception {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	out_of_bounds(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		throw()
		: exception(ns, clazz, method, file, line, "out_of_bounds",
			message) { };

	/**	\brief Virtual destructor
	 **/
	virtual ~out_of_bounds() throw() { };

	//@}
};

/**	\brief Throws an exception with a given error message
 **/
void throw_exc(const char *clazz, const char *method, const char *error)
	throw(exception);

} // namespace libtensor

#endif // LIBTENSOR_EXCEPTION_H

