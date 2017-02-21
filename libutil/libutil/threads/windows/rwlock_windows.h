#ifndef LIBUTIL_RWLOCK_WINDOWS_H
#define LIBUTIL_RWLOCK_WINDOWS_H

#include <libutil/exceptions/util_exceptions.h>
#include <windows.h>

namespace libutil {

/** \brief Windows implementation of the read-write lock

    Generic Windows doesn't assume the availability of SRW locks,
    which are introduced since Vista. Mutexes are used instead.

    \sa rwlock_windows_srw
    \ingroup libutil_threads
 **/
class rwlock_windows {
public:
    typedef HANDLE rwlock_id_type; //!< Read-write lock handle

public:
    static void create(rwlock_id_type &id) {

        id = CreateMutex(NULL, FALSE, NULL);
    }

    static void destroy(const rwlock_id_type &id) {

        CloseHandle(id);
    }

    static void rdlock(const rwlock_id_type &id) {

        lock(id);
    }

    static void wrlock(const rwlock_id_type &id) {

        lock(id);
    }

    static void unlock(const rwlock_id_type &id) {

        if(!ReleaseMutex(id)) {
            throw threads_exception("rwlock_windows",
                    "unlock(const rwlock_id_type &)",
                    __FILE__, __LINE__, "Unlock failed.");
        }
    }

private:
    static void lock(const rwlock_id_type &id) {

        DWORD res = WaitForSingleObject(id, INFINITE);
        if(res != WAIT_OBJECT_0) {
            throw threads_exception("rwlock_windows",
                    "lock(const rwlock_id_type &)",
                    __FILE__, __LINE__, "Lock failed.");
        }
    }

};


} // namespace libutil

#endif // LIBUTIL_RWLOCK_WINDOWS_H
