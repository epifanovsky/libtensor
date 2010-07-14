#ifndef LIBTENSOR_WORKER_H
#define LIBTENSOR_WORKER_H

#include <libvmm/thread.h>

namespace libtensor {


class worker : public libvmm::thread {
public:
	/**	\brief Virtual destructor
	 **/
	virtual ~worker() { }

	/**	\brief Runs the worker
	 **/
	virtual void run();
};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_H
