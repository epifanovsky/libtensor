#ifndef LIBUTIL_SPINLOCK_POSIX_H
#define LIBUTIL_SPINLOCK_POSIX_H

#include <pthread.h>

namespace libutil {


/** \brief POSIX implementation of the spinlock

    \ingroup libutil_threads
 **/
class spinlock_posix {
public:
    typedef pthread_spinlock_t mutex_id_type; //!< Spinlock handle type

public:
    /** \brief Creates the spinlock, initializes spinlock id
        \param[out] id Spinlock id.
     **/
    static void create(mutex_id_type &id);

    /** \brief Destroys the spinlock
        \param id Spinlock id.
     **/
    static void destroy(mutex_id_type &id);

    /** \brief Waits until the spinlock becomes available and then locks it
        \param id Spinlock id.
     **/
    static void lock(mutex_id_type &id);

    /** \brief Unlocks the spinlock and makes it available to other threads
        \param id Spinlock id.
     **/
    static void unlock(mutex_id_type &id);

};


} // namespace libutil

#endif // LIBUTIL_SPINLOCK_POSIX_H
