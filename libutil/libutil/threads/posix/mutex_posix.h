#ifndef LIBUTIL_MUTEX_POSIX_H
#define LIBUTIL_MUTEX_POSIX_H

#include <pthread.h>

namespace libutil {


/** \brief POSIX implementation of the mutex

    \ingroup libutil_threads
 **/
class mutex_posix {
public:
    typedef pthread_mutex_t mutex_id_type; //!< Mutex handle type

public:
    /** \brief Creates the mutex, initializes mutex id
        \param[out] id Mutex id.
     **/
    static void create(mutex_id_type &id);

    /** \brief Destroys the mutex
        \param id Mutex id.
     **/
    static void destroy(mutex_id_type &id);

    /** \brief Waits until the mutex becomes available and then locks it
        \param id Mutex id.
     **/
    static void lock(mutex_id_type &id);

    /** \brief Unlocks the mutex and makes it available to other threads
        \param id Mutex id.
     **/
    static void unlock(mutex_id_type &id);

};


} // namespace libutil

#endif // LIBUTIL_MUTEX_POSIX_H
