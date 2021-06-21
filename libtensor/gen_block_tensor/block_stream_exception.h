#ifndef LIBTENSOR_BLOCK_STREAM_EXCEPTION_H
#define LIBTENSOR_BLOCK_STREAM_EXCEPTION_H

#include <libtensor/exception.h>

namespace libtensor {


/** \brief Exception indicating that a block stream is in an incorrect state

    \sa gen_block_stream_i

    \ingroup libtensor_gen_block_tensor
 **/
class block_stream_exception : public exception_base<block_stream_exception> {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates an exception
     **/
    block_stream_exception(const char *ns, const char *clazz,
        const char *method, const char *file, unsigned int line,
        const char *message) noexcept :
        exception_base<block_stream_exception>(ns, clazz, method, file, line,
            "block_stream_exception", message) { };

    /** \brief Virtual destructor
     **/
    virtual ~block_stream_exception() noexcept { };

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_STREAM_EXCEPTION_H
