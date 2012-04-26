#ifndef LIBTENSOR_BAD_DIMENSIONS_H
#define LIBTENSOR_BAD_DIMENSIONS_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating that a %tensor passed to a %tensor operation
        has incorrect %dimensions

    \ingroup libtensor_core_exc libtensor_tod
 **/
class bad_dimensions : public exception_base<bad_dimensions> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    bad_dimensions(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        throw() :
        exception_base<bad_dimensions>(ns, clazz, method, file, line,
            "bad_dimensions", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~bad_dimensions() throw() { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_BAD_DIMENSIONS_H
