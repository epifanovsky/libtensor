#ifndef LIBTENSOR_EXPR_EXCEPTION_H
#define LIBTENSOR_EXPR_EXCEPTION_H

#include <libtensor/exception.h>

namespace libtensor {
namespace expr {


/** \brief Exception indicating an error in an expression

    \ingroup libtensor_expr
 **/
class expr_exception : public exception_base<expr_exception> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    expr_exception(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        throw() :
        exception_base<expr_exception>(ns, clazz, method, file, line,
            "expr_exception", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~expr_exception() throw() { };

    //@}
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EXCEPTION_H
