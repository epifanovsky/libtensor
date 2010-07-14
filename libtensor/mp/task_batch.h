#ifndef LIBTENSOR_TASK_BATCH_H
#define LIBTENSOR_TASK_BATCH_H

#ifdef LIBTENSOR_MP
#include "task_batch_mp_base.h"
#else
#include "task_batch_sp_base.h"
#endif // LIBTENSOR_MP

namespace libtensor {

class task_batch :
#ifdef LIBTENSOR_MP
	public task_batch_mp_base
#else
	public task_batch_sp_base
#endif // LIBTENSOR_MP
{
};


} // namespace libtensor

#endif // LIBTENSOR_TASK_BATCH_H
