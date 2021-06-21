#ifndef LIBUTIL_UTIL_EXCEPTIONS_H
#define LIBUTIL_UTIL_EXCEPTIONS_H

#include "exception_base.h"

namespace libutil {


/**	\brief Generic exception

	\ingroup libutil_exceptions
 **/
class generic_exception : public exception_base<generic_exception> {
private:
	typedef exception_base<generic_exception> exception_base_t;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an exception
	 **/
	generic_exception(const char *ns, const char *clazz, const char *method,
		const char *file, unsigned int line, const char *message)
		noexcept :
		exception_base_t(ns, clazz, method, file, line, "generic_exception",
			message) { };

	/**	\brief Creates an exception that originates in libvmm
	 **/
	generic_exception(const char *clazz, const char *method, const char *file,
		unsigned int line, const char *message) noexcept :
		exception_base_t("libutil", clazz, method, file, line,
			"generic_exception", message) { };

	/**	\brief Copy constructor
	 **/
	generic_exception(const generic_exception &e) noexcept :
	    exception_base_t(e) { }

	/**	\brief Virtual destructor
	 **/
	virtual ~generic_exception() noexcept { };

	//@}

};

/** \brief  Exception indicating an error in thread-related codes

    \ingroup libutil_exceptions
 **/
class threads_exception : public exception_base<threads_exception> {
private:
    typedef exception_base<threads_exception> exception_base_t;

public:
    //! \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    threads_exception(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        noexcept :
        exception_base_t(ns, clazz, method, file, line, "threads_exception",
            message) { };

    /** \brief Creates an exception that originates in libvmm
     **/
    threads_exception(const char *clazz, const char *method, const char *file,
        unsigned int line, const char *message) noexcept :
        exception_base_t("libutil", clazz, method, file, line,
            "threads_exception", message) { };

    /** \brief Copy constructor
     **/
    threads_exception(const threads_exception &e) noexcept :
        exception_base_t(e) { }

    /** \brief Virtual destructor
     **/
    virtual ~threads_exception() noexcept { };

    //@}

};


/** \brief Exception indicating an error in timings

    \ingroup libutil_exceptions
 **/
class timings_exception : public exception_base<timings_exception> {
private:
    typedef exception_base<timings_exception> exception_base_t;

public:
    //! \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    timings_exception(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        noexcept :
        exception_base_t(ns, clazz, method, file, line, "timings_exception",
            message) { };

    /** \brief Creates an exception that originates in libvmm
     **/
    timings_exception(const char *clazz, const char *method, const char *file,
        unsigned int line, const char *message) noexcept :
        exception_base_t("libutil", clazz, method, file, line,
            "timings_exception", message) { };

    /** \brief Copy constructor
     **/
    timings_exception(const timings_exception &e) noexcept :
        exception_base_t(e) { }

    /** \brief Virtual destructor
     **/
    virtual ~timings_exception() noexcept { };

    //@}

};


} // namespace libutil

#endif // LIBUTIL_UTIL_EXCEPTIONS_H
