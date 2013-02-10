#ifndef LIBTENSOR_CTF_ERROR_H
#define LIBTENSOR_CTF_ERROR_H

#include "../exception.h"

namespace libtensor {


/** \brief Exception indicating an error in Cyclops Tensor Framework

    \ingroup libtensor_ctf_dense_tensor
 **/
class ctf_error : public exception_base<ctf_error> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    ctf_error(const char *ns, const char *clazz, const char *method,
        const char *file, unsigned int line, const char *message)
        throw() :
        exception_base<ctf_error>(ns, clazz, method, file, line,
            "ctf_error", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~ctf_error() throw() { };

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_CTF_ERROR_H
