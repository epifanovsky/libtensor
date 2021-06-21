#ifndef LIBTENSOR_NOT_IMPLEMENTED_H
#define LIBTENSOR_NOT_IMPLEMENTED_H

#include "exception.h"

namespace libtensor {


/** \brief Indicates that a requested method is not yet implemented

    \ingroup libtensor_core_exc
 **/
class not_implemented : public exception_base<not_implemented> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    not_implemented(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line) noexcept :
        exception_base<not_implemented>(ns, clazz, method, file, line,
            "not_implemented", "NIY") { };

    /** \brief Virtual destructor
     **/
    virtual ~not_implemented() noexcept { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_NOT_IMPLEMENTED_H
