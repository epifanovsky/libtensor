#ifndef LIBUTIL_TLS_H
#define LIBUTIL_TLS_H

namespace libutil {


/** \brief Thread local storage for objects of type T

    \ingroup libutil_threads
 **/
template<typename T>
class tls;


} // namespace libutil


#if defined(USE_BUILTIN_TLS)

#include "builtin/tls_builtin.h"
namespace libutil {
template<typename T> class tls : public tls_builtin<T> { };
} // namespace libutil


#elif defined(USE_PTHREADS)

#include "posix/tls_posix.h"
namespace libutil {
template<typename T> class tls : public tls_posix<T> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS)

#include "windows/tls_windows.h"
namespace libutil {
template<typename T> class tls : public tls_windows<T> { };
} // namespace libutil


#else
#error Please read https://download.q-chem.com/trac/wiki/CompileCcman

#endif


#endif // LIBUTIL_TLS_H
