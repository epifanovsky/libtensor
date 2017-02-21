#ifndef LIBUTIL_TLS_POSIX_H
#define LIBUTIL_TLS_POSIX_H

#include <pthread.h>
#include <libutil/singleton.h>

namespace libutil {


/** \brief POSIX implementation of the thread local storage (base class)

    \ingroup libutil_threads
 **/
class tls_posix_base {
private:
    pthread_key_t m_key; //!< TLS key

public:
    /** \brief Default constructor
        \param destructor Pointer to the destructor function.
     **/
    tls_posix_base(void (*destructor)(void*));

    /** \brief Virtual destructor
     **/
    virtual ~tls_posix_base();

protected:
    /** \brief Sets thread-specific pointer
     **/
    void set_ptr(void *ptr);

    /** \brief Returns thread-specific pointer
     **/
    void *get_ptr();

};


/** \brief POSIX implementation of the thread local storage
    \tparam T Type of the object stored in TLS

    \ingroup libutil_threads
 **/
template<typename T>
class tls_posix : public tls_posix_base, public singleton< tls_posix<T> > {
    friend class singleton< tls_posix<T> >;

protected:
    /** \brief Protected singleton constructor
     **/
    tls_posix() : tls_posix_base(destructor) { }

public:
    /** \brief Returns the object stored in TLS
     **/
    T &get() {
        void *ptr = get_ptr();
        T *t = static_cast<T*>(ptr);
        if(t == 0) set_ptr(t = new T());
        return *t;
    }

private:
    static void destructor(void *ptr) {
        delete static_cast<T*>(ptr);
        tls_posix<T>::get_instance().set_ptr(0);
    }

};


} // namespace libutil

#endif // LIBUTIL_TLS_POSIX_H
