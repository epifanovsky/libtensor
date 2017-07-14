#ifndef LIBUTIL_TASK_I_H
#define LIBUTIL_TASK_I_H

namespace libutil {


/** \brief Task interface

    \ingroup libutil_thread_pool
 **/
class task_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~task_i() { }

    /** \brief Returns the cost of the task in arbitrary units
     **/
    virtual unsigned long get_cost() const = 0;

    /** \brief Performs the task
     **/
    virtual void perform() = 0;

};


} // namespace libutil

#endif // LIBUTIL_TASK_I_H
