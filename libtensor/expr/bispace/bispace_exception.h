#ifndef LIBTENSOR_EXPR_BISPACE_EXCEPTION_H
#define LIBTENSOR_EXPR_BISPACE_EXCEPTION_H

#include <libtensor/exception.h>

namespace libtensor {
namespace expr {


/** \brief Exception indicating an error in a bispace expression

    \ingroup libtensor_expr_bispace
 **/
class bispace_exception : public exception_base<bispace_exception> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    bispace_exception(const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message) throw() :
        exception_base<bispace_exception>("libtensor::expr", clazz, method,
            file, line, "bispace_exception", message)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~bispace_exception() throw() { }

    //@}

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_BISPACE_EXCEPTION_H
