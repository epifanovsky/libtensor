#ifndef LIBUTIL_UNKNOWN_EXCEPTION_H
#define LIBUTIL_UNKNOWN_EXCEPTION_H

#include <exception>
#include <libutil/exceptions/rethrowable_i.h>

namespace libutil {


/** \brief Wrapper for an unknown exception

    \ingroup libutil_thread_pool
 **/
class unknown_exception : public std::exception, public rethrowable_i {
private:
    static const char *k_what;

public:
    /** \brief Virtual destructor
     **/
    virtual ~unknown_exception() noexcept;

    /** \brief Returns the type of exception
     **/
    virtual const char *what() const noexcept;

    /** \brief Clones this exception using operator new
     **/
    virtual rethrowable_i *clone() const noexcept;

    /** \brief Rethrows this exception
     **/
    virtual void rethrow() const;

};


} // namespace libutil

#endif // LIBUTIL_UNKNOWN_EXCEPTION_H
