#include "tls_posix.h"

namespace libutil {


tls_posix_base::tls_posix_base(void (*destructor)(void*)) {

    pthread_key_create(&m_key, destructor);
}


tls_posix_base::~tls_posix_base() {

    pthread_key_delete(m_key);
}


void tls_posix_base::set_ptr(void *ptr) {

    pthread_setspecific(m_key, ptr);
}


void *tls_posix_base::get_ptr() {

    return pthread_getspecific(m_key);
}


} // namespace libutil
