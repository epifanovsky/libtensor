#ifndef LIBUTIL_THREAD_BASE_H
#define LIBUTIL_THREAD_BASE_H

#include "thread_i.h"

namespace libutil {


/** \brief Base class for the thread
    \tparam Impl Implementation.

    \ingroup libutil_threads
 **/
template<typename Impl>
class thread_base : public thread_i {
private:
    typename Impl::thread_id_type m_id; //!< Thread id
    bool m_valid_id; //!< Whether the thread id is valid

public:
    /** \brief Default constructor
     **/
    thread_base() {
        m_valid_id = false;
    }

    /** \brief Virtual destructor
     **/
    virtual ~thread_base() {
        if(m_valid_id) Impl::destroy(m_id);
    }

    /** \brief Starts the thread
     **/
    void start() {
        m_id = Impl::create(this);
        m_valid_id = true;
    }

    /** \brief Waits for the thread to finish
     **/
    void join() {
        Impl::join(m_id);
    }

private:
    thread_base(const thread_base<Impl>&);
    const thread_base<Impl> &operator=(const thread_base<Impl>&);

};


} // namespace libutil

#endif // LIBUTIL_THREAD_BASE_H
