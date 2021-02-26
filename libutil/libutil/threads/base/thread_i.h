#ifndef LIBUTIL_THREAD_I_H
#define LIBUTIL_THREAD_I_H

namespace libutil {


/** \brief Thread interface

    \ingroup libutil_threads
 **/
class thread_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~thread_i() { }

    /** \brief Runs the process in the thread
     **/
    virtual void run() = 0;

};


} // namespace libutil

#endif // LIBUTIL_THREAD_I_H
