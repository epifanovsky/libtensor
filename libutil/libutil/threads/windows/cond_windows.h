#ifndef LIBUTIL_COND_WINDOWS_H
#define LIBUTIL_COND_WINDOWS_H

#include <libutil/exceptions/util_exceptions.h>
#include <windows.h>

namespace libutil {


/** \brief Windows implementation of the conditional variable

    \ingroup libutil_threads
 **/
class cond_windows {
public:
    typedef struct {
        bool m_waiting;
        volatile bool m_sig;
        HANDLE m_event;
        CRITICAL_SECTION m_wait_lock;
        CRITICAL_SECTION m_ext_lock;
    } cond_id_type;

public:
    static void create(cond_id_type &id) {

        id.m_waiting = false;
        id.m_sig = false;
        id.m_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        InitializeCriticalSection(&id.m_wait_lock);
        InitializeCriticalSection(&id.m_ext_lock);
        EnterCriticalSection(&id.m_ext_lock);
    }

    static void destroy(cond_id_type &id) {

        LeaveCriticalSection(&id.m_ext_lock);
        DeleteCriticalSection(&id.m_wait_lock);
        DeleteCriticalSection(&id.m_ext_lock);
        CloseHandle(id.m_event);
    }

    static void wait(cond_id_type &id) {

        bool err = false;
        EnterCriticalSection(&id.m_wait_lock);
        if(id.m_waiting) {
            err = true;
        } else {
            if(id.m_sig) {
                id.m_sig = false;
                LeaveCriticalSection(&id.m_wait_lock);
                return;
            }
            id.m_waiting = true;
        }
        LeaveCriticalSection(&id.m_wait_lock);
        if(err) {
            throw threads_exception("cond_windows", "wait(cond_id_type &)",
                    __FILE__, __LINE__, "Double wait.");
        }

        LeaveCriticalSection(&id.m_ext_lock);
        int rc = WaitForSingleObject(id.m_event, INFINITE);
        if(rc != WAIT_OBJECT_0) {
            throw threads_exception("cond_windows", "wait(cond_id_type &)",
                    __FILE__, __LINE__, "Error.");
        }

        EnterCriticalSection(&id.m_wait_lock);
        id.m_waiting = false;
        LeaveCriticalSection(&id.m_wait_lock);

        EnterCriticalSection(&id.m_ext_lock);
    }

    static void signal(cond_id_type &id) {

        EnterCriticalSection(&id.m_wait_lock);
        bool waiting = id.m_waiting;
        if(!waiting) id.m_sig = true;
        LeaveCriticalSection(&id.m_wait_lock);
        if(waiting) {
            SetEvent(id.m_event);
        }
    }

};


} // namespace libvmm

#endif // LIBVMM_COND_WINDOWS_H
