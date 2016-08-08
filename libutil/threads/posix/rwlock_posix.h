#ifndef LIBUTIL_RWLOCK_POSIX_H
#define LIBUTIL_RWLOCK_POSIX_H

#include <pthread.h>

namespace libutil {


/** \brief POSIX implementation of the read-write lock

    \ingroup libutil_threads
 **/
class rwlock_posix {
public:
    typedef pthread_rwlock_t rwlock_id_type; //!< Lock handle type

public:
    /** \brief Creates the read-write lock, initializes lock id
        \param[out] id Lock id.
     **/
    static void create(rwlock_id_type &id);

    /** \brief Destroys the read-write lock
        \param id Lock id.
     **/
    static void destroy(rwlock_id_type &id);

    /** \brief Locks for reading only
        \param id Lock id.
     **/
    static void rdlock(rwlock_id_type &id);

    /** \brief Locks for writing (exclusive)
        \param id Lock id.
     **/
    static void wrlock(rwlock_id_type &id);

    /** \brief Unlocks the read-write lock
        \param id Lock id.
     **/
    static void unlock(rwlock_id_type &id);

};


} // namespace libutil

#endif // LIBUTIL_RWLOCK_POSIX_H
