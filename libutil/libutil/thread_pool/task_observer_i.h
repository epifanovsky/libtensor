#ifndef LIBUTIL_TASK_OBSERVER_I_H
#define LIBUTIL_TASK_OBSERVER_I_H

#include "task_i.h"

namespace libutil {


/** \brief Task observer (event handler) interface

    \ingroup libutil_thread_pool
 **/
class task_observer_i {
public:
    virtual void notify_start_task(task_i *t) = 0;
    virtual void notify_finish_task(task_i *t) = 0;

};


} // namespace libutil

#endif // LIBUTIL_TASK_OBSERVER_I_H

