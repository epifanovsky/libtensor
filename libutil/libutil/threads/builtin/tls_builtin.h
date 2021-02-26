#ifndef LIBUTIL_TLS_BUILTIN_H
#define LIBUTIL_TLS_BUILTIN_H

#include <libutil/singleton.h>

namespace libutil {


/** \brief Built-in C++ implementation of the thread local storage
    \tparam T Type of the object stored in TLS

    \ingroup libutil_threads
 **/
template<typename T>
class tls_builtin : public singleton< tls_builtin<T> > {
    friend class singleton< tls_builtin<T> >;

private:
#if defined(HAVE_CPP_DECLSPEC_THREAD)
    __declspec(thread) static T *m_t;
#elif defined(HAVE_GCC_THREAD_LOCAL)
    static __thread T *m_t;
#else
#error No built-in TLS specified
#endif

protected:
    /** \brief Protected singleton constructor
     **/
    tls_builtin() { }

public:
    /** \brief Destructor
     **/
    ~tls_builtin() {
        delete m_t;
        m_t = 0;
    }

    /** \brief Returns the object stored in TLS
     **/
    T &get() {
        if(m_t == 0) m_t = new T();
        return *m_t;
    }

};


#if defined(HAVE_CPP_DECLSPEC_THREAD)
template<typename T>
T *tls_builtin<T>::m_t = 0;
#elif defined(HAVE_GCC_THREAD_LOCAL)
template<typename T>
__thread T *tls_builtin<T>::m_t = 0;
#endif


} // namespace libutil

#endif // LIBUTIL_TLS_BUILTIN_H
