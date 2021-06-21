#ifndef LIBTENSOR_OUT_OF_BOUNDS_H
#define LIBTENSOR_OUT_OF_BOUNDS_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating that an %index is out of bounds

    \ingroup libtensor_core_exc
 **/
class out_of_bounds : public exception_base<out_of_bounds> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    out_of_bounds(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        noexcept
        : exception_base<out_of_bounds>(ns, clazz, method, file, line,
            "out_of_bounds", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~out_of_bounds() noexcept { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_OUT_OF_BOUNDS_H

