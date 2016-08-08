#ifndef LIBUTIL_COND_BASE_H
#define LIBUTIL_COND_BASE_H

namespace libutil {


/** \brief Conditional variable base class
    \tparam Impl Implementation.

    \ingroup libutil_threads
 **/
template<typename Impl>
class cond_base {
private:
    typename Impl::cond_id_type m_id; //!< Conditional variable ID

public:
    /**	\brief Default constructor
     **/
    cond_base() {
        Impl::create(m_id);
    }

    /**	\brief Virtual destructor
     **/
    virtual ~cond_base() {
        Impl::destroy(m_id);
    }

    /**	\brief Waits until the condition has been turned on, then
            resets the condition
     **/
    void wait() {
        Impl::wait(m_id);
    }

    /**	\brief Turns on the condition thereby waking the waiting thread
     **/
    void signal() {
        Impl::signal(m_id);
    }

    /**	\brief Broadcasts a signal (wakes up all waiting threads)
     **/
    void broadcast() {
        Impl::broadcast(m_id);
    }

private:
    cond_base(const cond_base<Impl>&);
    const cond_base<Impl> &operator=(const cond_base<Impl>&);

};


} // namespace libutil

#endif // LIBUTIL_COND_BASE_H
