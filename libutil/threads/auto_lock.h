#ifndef LIBUTIL_AUTO_LOCK_H
#define LIBUTIL_AUTO_LOCK_H

namespace libutil {


/** \brief Automatically obtains a lock upon construction and releases it upon
        destruction
    \tparam Mutex Mutex type.

    \sa mutex, spinlock

    \ingroup libutil_threads
 **/
template<typename Mutex>
class auto_lock {
private:
    Mutex &m_mtx; //!< Underlying mutex

public:
    auto_lock(Mutex &mtx) : m_mtx(mtx) {
        m_mtx.lock();
    }

    ~auto_lock() noexcept {
        m_mtx.unlock();
    }

};


} // namespace libutil

#endif // LIBUTIL_AUTO_LOCK_H
