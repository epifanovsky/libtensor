#ifndef LIBTENSOR_EXPR_EXCEPTION_H
#define LIBTENSOR_EXPR_EXCEPTION_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating an error in an expression

    \ingroup libtensor_expr
 **/
class expr_exception : public exception_base<expr_exception> {
public:
    /**	\brief Creates an exception
     **/
    expr_exception(const char *clazz, const char *method, const char *file,
        unsigned int line, const char *message) throw() :
        exception_base<expr_exception>(g_ns, clazz, method, file, line,
            "expr_exception", message) { };

    /**	\brief Virtual destructor
     **/
    virtual ~expr_exception() throw() { };

};


} // namespace libtensor

#endif // LIBTENSOR_EXPR_EXCEPTION_H
