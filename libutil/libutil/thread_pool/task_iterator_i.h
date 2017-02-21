#ifndef LIBUTIL_TASK_ITERATOR_I_H
#define LIBUTIL_TASK_ITERATOR_I_H

#include "task_i.h"

namespace libutil {


/** \brief Task iterator

    \ingroup libutil_thread_pool
 **/
class task_iterator_i {
public:
    /** \brief Returns true if there are more tasks, false otherwise
     **/
    virtual bool has_more() const = 0;

    /** \brief Returns pointer to the next task
     **/
    virtual task_i *get_next() = 0;

};


} // namespace libutil

#endif // LIBUTIL_TASK_ITERATOR_I_H

