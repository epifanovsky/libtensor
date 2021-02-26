#ifndef LIBUTIL_THREAD_WINDOWS_H
#define LIBUTIL_THREAD_WINDOWS_H

#include <libutil/exceptions/util_exceptions.h>
#include <windows.h>
#include "tls_windows.h"
#include "../base/thread_i.h"

namespace libutil {


/** \brief Windows implementation of the thread

    \ingroup libutil_threads
 **/
class thread_windows {
public:
    typedef HANDLE thread_id_type; //!< Thread handle

public:
    static thread_id_type create(thread_i *thr) {
        thread_id_type id = CreateThread(NULL, 0, thread_main, (void*)thr, 0,
            NULL);
        if(id == NULL) {
            throw threads_exception("thread_windows", "create(thread_i *)",
                    __FILE__, __LINE__, "CreateThread failed.");
        }
        return id;
    }

    static void destroy(thread_id_type id) {
        CloseHandle(id);
    }

    static void join(thread_id_type id) {
        DWORD rc = WaitForSingleObject(id, INFINITE);
        if(rc != WAIT_OBJECT_0) {
            throw threads_exception("thread_windows", "join(thread_id_type)",
                    __FILE__, __LINE__, "Invalid wait result.");
        }
    }

    static DWORD WINAPI thread_main(LPVOID param) {
        thread_i *thr = (thread_i*)param;
        int rc = 0;
        try {
            thr->run();
        } catch(...) {
            rc = -1;
        }
        tls_windows_destructor_list::invoke_destructors();
        return rc;
    }

};


} // namespace libutil

#endif // LIBUTIL_THREAD_WINDOWS_H
