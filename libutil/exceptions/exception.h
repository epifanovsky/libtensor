#ifndef LIBUTIL_EXCEPTION_H
#define LIBUTIL_EXCEPTION_H

#include <exception>
#include <memory>
#include "rethrowable_i.h"

namespace libutil {


/**	\brief Base exception class

	\ingroup libutil_exceptions
 **/
class exception : public std::exception, public rethrowable_i {
private:
    char m_ns[128]; //!< Namespace name
    char m_clazz[128]; //!< Class name
    char m_method[128]; //!< Method name
    char m_file[128]; //!< Source file name
    unsigned int m_line; //!< Line number
    char m_type[128]; //!< Exception type
    char m_message[256]; //!< Exception message
    char m_what[1024]; //!< Composed message available via what()

public:
	//!	\name Construction and destruction
	//@{

    /** \brief Creates an %exception using full details
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
        const char *message) throw() {

        init(ns, clazz, method, file, line, type, message);
    }

    /** \brief Copy constructor
        \param e Other exception
     **/
    exception(const exception &e) {
        init(e.m_ns, e.m_clazz, e.m_method,
                e.m_file, e.m_line, e.m_type, e.m_message);
    }

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

private:
	void init(const char *ns, const char *clazz, const char *method,
	        const char *file, unsigned int line, const char *type,
	        const char *message) throw();
};


} // namespace libutil

#endif // LIBUTIL_EXCEPTION_H

