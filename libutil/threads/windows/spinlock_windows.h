#ifndef LIBUTIL_SPINLOCK_WINDOWS_H
#define LIBUTIL_SPINLOCK_WINDOWS_H

#include <windows.h>

namespace libutil {


/** \brief Windown implementation of the spinlock

    \ingroup libutil_threads
 **/
class spinlock_windows {
public:
    typedef CRITICAL_SECTION mutex_id_type;

public:
    static void create(mutex_id_type &id) {
        InitializeCriticalSection(&id);
    }

    static void destroy(mutex_id_type &id) {
        DeleteCriticalSection(&id);
    }

    static void lock(mutex_id_type &id) {
        EnterCriticalSection(&id);
    }

    static void unlock(mutex_id_type &id) {
        LeaveCriticalSection(&id);
    }

};


} // namespace libutil

#endif // LIBUTIL_SPINLOCK_WINDOWS_H
