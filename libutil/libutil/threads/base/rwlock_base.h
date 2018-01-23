#ifndef LIBUTIL_RWLOCK_BASE_H
#define LIBUTIL_RWLOCK_BASE_H

namespace libutil {


/** \brief Base class for the read-write lock
    \tparam Impl Implementation.

    \ingroup libutil_threads
 **/
template<typename Impl>
class rwlock_base {
private:
    typename Impl::rwlock_id_type m_id; //!< Lock ID

public:
    /** \brief Default constructor
     **/
    rwlock_base() {
        Impl::create(m_id);
    }

    /** \brief Virtual destructor
     **/
    virtual ~rwlock_base() {
        Impl::destroy(m_id);
    }

    /** \brief Obtains a read-only lock
     **/
    void rdlock() {
        Impl::rdlock(m_id);
    }

    /** \brief Obtains a write lock
     **/
    void wrlock() {
        Impl::wrlock(m_id);
    }

    /** \brief Releases the read-write lock
     **/
    void unlock() {
        Impl::unlock(m_id);
    }

private:
    rwlock_base(const rwlock_base<Impl>&);
    const rwlock_base<Impl> &operator=(const rwlock_base<Impl>&);

};


} // namespace libutil

#endif // LIBUTIL_RWLOCK_BASE_H
