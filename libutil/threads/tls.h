#ifndef LIBUTIL_TLS_H
#define LIBUTIL_TLS_H
#include "builtin/tls_builtin.h"

namespace libutil {


/** \brief Thread local storage for objects of type T

    \ingroup libutil_threads
 **/
template<typename T> class tls : public tls_builtin<T> { };


} // namespace libutil

#endif // LIBUTIL_TLS_H
