#include <libutil/exceptions/util_exceptions.h>
#include "thread_posix.h"

namespace libutil {


thread_posix::thread_id_type thread_posix::create(thread_i *thr) {

    const size_t min_stacksz = 2 * 1024 * 1024; // 2 megabytes
    size_t stacksz = 0;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    //  Set a minimum stack size of min_stacksz
    pthread_attr_getstacksize(&attr, &stacksz);
    if(stacksz < min_stacksz) {
        pthread_attr_setstacksize(&attr, min_stacksz);
    }

    thread_id_type id;
    int rc = pthread_create(&id, &attr, thread_main, (void*)thr);
    pthread_attr_destroy(&attr);
    if(rc) {
        throw threads_exception("thread_posix", "create(thread_i *)",
                __FILE__, __LINE__, "Operation failed.");
    }
    return id;
}


void thread_posix::destroy(const thread_id_type &id) {

}


void thread_posix::join(const thread_id_type &id) {

    void *status;
    int rc = pthread_join(id, &status);
    if(rc) {
        throw threads_exception("thread_posix", "join(const thread_id_type &)",
                __FILE__, __LINE__, "Operation failed.");
    }
}


void *thread_posix::thread_main(void *param) {

    thread_i *thr = (thread_i*)param;
    size_t rc = 0;
    try {
        thr->run();
    } catch(...) {
        rc = (size_t)(-1);
    }
    return (void*)rc;
}


} // namespace libutil
