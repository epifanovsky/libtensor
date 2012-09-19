#ifndef LIBTENSOR_AUTO_RWLOCK_H
#define LIBTENSOR_AUTO_RWLOCK_H

#include <libutil/threads/rwlock.h>
#include <libutil/threads/mutex.h>

namespace libtensor {


/** \brief Automatic scope read-write lock

    This lock protects its scope and provides a multiple reader-exclusive
    writer facility.

    \ingroup libtensor_gen_block_tensor
 **/
class auto_rwlock {
private:
    libutil::rwlock *m_lock;
    libutil::mutex *m_locku;
    bool m_wr;

public:
    /** \brief Initializes the lock
     **/
    auto_rwlock(libutil::rwlock *lock, libutil::mutex *locku, bool wr);

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
