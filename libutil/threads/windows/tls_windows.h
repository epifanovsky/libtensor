#ifndef LIBUTIL_TLS_WINDOWS_H
#define LIBUTIL_TLS_WINDOWS_H

#include <vector>
#include <windows.h>
#include <libutil/singleton.h>
#include "../auto_lock.h"
#include "../mutex.h"

namespace libutil {


class tls_windows_destructor_list {
public:
    typedef void (*destructor_fn)(void);

private:
    static mutex m_lock;
    static std::vector<destructor_fn> m_lst;

public:
    static void add_destructor(destructor_fn d) {
        auto_lock<mutex> lock(m_lock);
        m_lst.push_back(d);
    }

    static void invoke_destructors() {
        auto_lock<mutex> lock(m_lock);
        for(std::vector<destructor_fn>::iterator i = m_lst.begin();
            i != m_lst.end(); ++i) (**i)();
    }

};


/** \brief Windows implementation of the thread local storage

    \ingroup libutil_threads
 **/
template<typename T>
class tls_windows : public singleton< tls_windows<T> > {
    friend class singleton< tls_windows<T> >;

private:
    DWORD m_key; //!< TLS key

protected:
    /** \brief Protected singleton constructor
     **/
    tls_windows() {
        m_key = TlsAlloc();
    }

public:
    /** \brief Virtual destructor
     **/
    ~tls_windows() {
        TlsFree(m_key);
    }

public:
    T &get() {
        void *ptr = TlsGetValue(m_key);
        T *t = static_cast<T*>(ptr);
        if(t == 0) {
            TlsSetValue(m_key, t = new T());
            tls_windows_destructor_list::add_destructor(&destructor);
        }
        return *t;
    }

private:
    static void destructor() {
        void *ptr = TlsGetValue(tls_windows<T>::get_instance().m_key);
        delete static_cast<T*>(ptr);
        TlsSetValue(tls_windows<T>::get_instance().m_key, (void*)0);
    }

};


} // namespace libutil

#endif // LIBUTIL_TLS_WINDOWS_H
