#ifndef LIBUTIL_SPINLOCK_MACOS_H
#define LIBUTIL_SPINLOCK_MACOS_H

#include <libkern/OSAtomic.h>

namespace libutil {


/** \brief Mac OS implementation of the spinlock

    \ingroup libutil_threads
 **/
class spinlock_macos {
public:
    typedef OSSpinLock mutex_id_type;

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

#endif // LIBUTIL_SPINLOCK_MACOS_H
