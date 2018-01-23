#ifndef LIBUTIL_THREAD_POSIX_H
#define LIBUTIL_THREAD_POSIX_H

#include <pthread.h>
#include "../base/thread_i.h"

namespace libutil {


/** \brief POSIX implementation of the thread

    \ingroup libutil_threads
 **/
class thread_posix {
public:
    typedef pthread_t thread_id_type; //!< Thread handle

public:
    /** \brief Initializes and runs a thread
        \param thr Thread.
        \return Thread id.
     **/
    static thread_id_type create(thread_i *thr);

    /** \brief Destroys a thread
        \param id Thread id.
     **/
    static void destroy(const thread_id_type &id);

    /** \brief Waits for a thread to finish
        \param id Thread id.
     **/
    static void join(const thread_id_type &id);

private:
    /** \brief Thread's main function
     **/
    static void *thread_main(void *param);

};


} // namespace libutil

#endif // LIBUTIL_THREAD_POSIX_H
