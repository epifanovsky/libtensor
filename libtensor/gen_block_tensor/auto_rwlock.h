#ifndef LIBTENSOR_AUTO_RWLOCK_H
#define LIBTENSOR_AUTO_RWLOCK_H

#include <libutil/threads/rwlock.h>

namespace libtensor {


/** \brief Automatic scope read-write lock

    This lock protects its scope and provides a multiple reader-exclusive
    writer facility.

    Upon creation, auto_rwlock acquires a read-only lock from the underlying
    libutil::rwlock object. It can be later upgrade()ed to a read-write lock,
    which in turn can be downgrade()ed back to the read-only lock.

    \ingroup libtensor_gen_block_tensor
 **/
class auto_rwlock {
private:
    libutil::rwlock &m_lock;
    bool m_wr;

public:
    /** \brief Initializes the lock
     **/
    auto_rwlock(libutil::rwlock &lock);

    /** \brief Destroys the lock
     **/
    ~auto_rwlock();

    /** \brief Upgrades a read-lock to a write-lock
     **/
    void upgrade();

    /** \brief Downgrades a write-lock to a read-lock
     **/
    void downgrade();

};


} // namespace libtensor

#endif // LIBTENSOR_AUTO_RWLOCK_H
