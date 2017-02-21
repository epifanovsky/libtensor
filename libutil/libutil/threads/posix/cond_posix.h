#ifndef LIBUTIL_COND_POSIX_H
#define LIBUTIL_COND_POSIX_H

#include <pthread.h>

namespace libutil {


/** \brief POSIX implementation of the conditional variable

    \ingroup libutil_threads
 **/
class cond_posix {
public:
    typedef struct {
        pthread_mutex_t m_mtx;
        pthread_cond_t m_cond;
        volatile bool m_sig;
    } cond_id_type;

public:
    /** \brief Creates the conditional, initializes its id
        \param[out] id Conditional id.
     **/
    static void create(cond_id_type &id);

    /** \brief Destroys the conditional
        \param id Conditional id.
     **/
    static void destroy(cond_id_type &id);

    /** \brief Waits for the condition on the variable to be met
        \param id Conditional id.
     **/
    static void wait(cond_id_type &id);

    /** \brief Signals that the conditional has been met
        \param id Conditional id.
     **/
    static void signal(cond_id_type &id);

    /** \brief Wakes up all the threads that are waiting on the conditional
        \param id Conditional id.
     **/
    static void broadcast(cond_id_type &id);

};


} // namespace libutil

#endif // LIBUTIL_COND_POSIX_H
