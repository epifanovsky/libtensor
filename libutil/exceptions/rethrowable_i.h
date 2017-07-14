#ifndef LIBUTIL_RETHROWABLE_I_H
#define LIBUTIL_RETHROWABLE_I_H

namespace libutil {


/** \brief Rethrowable exception interface

    \ingroup libutil_exceptions
 **/
class rethrowable_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~rethrowable_i() throw();

    /** \brief Clones this exception using operator new
     **/
    virtual rethrowable_i *clone() const throw() = 0;

    /** \brief Rethrows this exception
     **/
    virtual void rethrow() const = 0;

};


} // namespace libutil

#endif // LIBUTIL_RETHROWABLE_I_H
