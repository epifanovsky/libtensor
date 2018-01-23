#ifndef LIBUTIL_MUTEX_BASE_H
#define LIBUTIL_MUTEX_BASE_H

namespace libutil {


/** \brief Base class for the mutex
    \tparam Impl Implementation.

    \ingroup libutil_threads
 **/
template<typename Impl>
class mutex_base {
private:
    typename Impl::mutex_id_type m_id; //!< Mutex ID

public:
    /** \brief Default constructor
     **/
    mutex_base() {
        Impl::create(m_id);
    }

    /** \brief Virtual destructor
     **/
    virtual ~mutex_base() {
        Impl::destroy(m_id);
    }

    /** \brief Obtains a lock on the mutex
     **/
    void lock() {
        Impl::lock(m_id);
    }

    /** \brief Releases the mutex
     **/
    void unlock() {
        Impl::unlock(m_id);
    }

private:
    mutex_base(const mutex_base<Impl>&);
    const mutex_base<Impl> &operator=(const mutex_base<Impl>&);

};


} // namespace libutil

#endif // LIBUTIL_MUTEX_BASE_H
