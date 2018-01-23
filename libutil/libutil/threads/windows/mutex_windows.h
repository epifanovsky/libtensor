#ifndef LIBUTIL_MUTEX_WINDOWS_H
#define LIBUTIL_MUTEX_WINDOWS_H

#include <libutil/exceptions/util_exceptions.h>
#include <windows.h>

namespace libutil {


/** \brief Windows implementation of the mutex

    \ingroup libutil_threads
 **/
class mutex_windows {
public:
    typedef HANDLE mutex_id_type; //!< Mutex handle type

public:
    static void create(mutex_id_type &id) {

        id = CreateMutex(NULL, FALSE, NULL);
    }

    static void destroy(const mutex_id_type &id) {

        CloseHandle(id);
    }

    static void lock(const mutex_id_type &id) {

        DWORD res = WaitForSingleObject(id, INFINITE);
        if(res != WAIT_OBJECT_0) {
            throw threads_exception("mutex_windows",
                    "lock(const mutex_id_type &)", __FILE__, __LINE__,
                    "Lock failed.");
        }
    }

    static void unlock(const mutex_id_type &id) {

        if(!ReleaseMutex(id)) {
            throw threads_exception("mutex_windows",
                    "unlock(const mutex_id_type &)", __FILE__, __LINE__,
                    "Unlock failed.");
        }
    }

};


} // namespace libutil

#endif // LIBUTIL_MUTEX_WINDOWS_H
