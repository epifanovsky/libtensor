#ifndef LIBTENSOR_TASK_BATCH_SP_BASE_H
#define LIBTENSOR_TASK_BATCH_SP_BASE_H

#include "task_i.h"

namespace libtensor {


/**	\brief Task batch base class for single-processor execution

	\ingroup libtensor_mp
 **/
class task_batch_sp_base {
public:
	void push(task_i &task) {
		task.perform();
	}

	void wait() {
	}
};


} // namespace libtensor

#endif // LIBTENSOR_TASK_BATCH_SP_BASE_H
