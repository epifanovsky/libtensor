#include "../auto_rwlock.h"

using libutil::mutex;
using libutil::rwlock;

namespace libtensor {


auto_rwlock::auto_rwlock(rwlock *lock, mutex *locku, bool wr) :
    m_lock(lock), m_locku(locku), m_wr(wr) {

    if(m_lock) {
        if(wr) m_lock->wrlock();
        else m_lock->rdlock();
    }
}


auto_rwlock::~auto_rwlock() {

    if(m_lock) {
        m_lock->unlock();
    }
}


void auto_rwlock::upgrade() {

    if(m_lock && !m_wr) {
        m_lock->unlock();
        m_lock->wrlock();
        m_wr = true;
    }
}


void auto_rwlock::downgrade() {

    if(m_lock && m_wr) {
        m_lock->unlock();
        m_lock->rdlock();
        m_wr = false;
    }
}


} // namespace libtensor
