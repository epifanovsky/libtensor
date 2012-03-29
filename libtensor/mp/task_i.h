#ifndef LIBTENSOR_TASK_I_H
#define LIBTENSOR_TASK_I_H

#include "../exception.h"
#include "cpu_pool.h"

namespace libtensor {


/** \brief Task interface

    \ingroup libtensor_mp
 **/
class task_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~task_i() { }

    /** \brief Executes the task on a pool of CPUs
     **/
    virtual void perform(cpu_pool &cpus) throw(exception) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_TASK_I_H
