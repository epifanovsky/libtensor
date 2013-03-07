#ifndef LIBTENSOR_CUDA_ERROR_H
#define LIBTENSOR_CUDA_ERROR_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating that a call to CUDA resulted in an error

    \ingroup libtensor_cuda
 **/
class cuda_error : public exception_base<cuda_error> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    cuda_error(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        throw() :
        exception_base<cuda_error>(ns, clazz, method, file, line,
            "cuda_error", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~cuda_error() throw() { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_ERROR_H
