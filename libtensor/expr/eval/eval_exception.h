#ifndef LIBTENSOR_EVAL_EXCEPTION_H
#define LIBTENSOR_EVAL_EXCEPTION_H

#include <libtensor/exception.h>

namespace libtensor {
namespace expr {


/** \brief Exception indicating an error during evaluation of expression

    \ingroup libtensor_expr_eval
 **/
class eval_exception : public exception_base<eval_exception> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    eval_exception(const char *file, unsigned int line, const char *ns,
        const char *clazz, const char *method, const char *message)
        throw() :
        exception_base<eval_exception>(ns, clazz, method, file, line,
            "eval_exception", message) { }

    /** \brief Virtual destructor
     **/
    virtual ~eval_exception() throw() { }

    //@}
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EVAL_EXCEPTION_H
