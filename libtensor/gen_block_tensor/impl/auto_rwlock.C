#include "../auto_rwlock.h"

namespace libtensor {


auto_rwlock::auto_rwlock(libutil::rwlock &lock) :
    m_lock(lock), m_wr(false) {

    m_lock.rdlock();
}


auto_rwlock::~auto_rwlock() {

    m_lock.unlock();
}


void auto_rwlock::upgrade() {

    if(!m_wr) {
        m_lock.unlock();
        m_lock.wrlock();
        m_wr = true;
    }
}


void auto_rwlock::downgrade() {

    if(m_wr) {
        m_lock.unlock();
        m_lock.rdlock();
        m_wr = false;
    }
}


} // namespace libtensor
