#ifndef LIBTENSOR_WORKER_H
#define LIBTENSOR_WORKER_H

#include <libvmm/cond.h>
#include <libvmm/thread.h>
#include "../timings.h"

namespace libtensor {


/**	\brief Worker thread

	\ingroup libtensor_mp
 **/
class worker : public libvmm::thread, public timings<worker> {
public:
	static const char *k_clazz; //!< Class name

private:
	libvmm::cond &m_started; //!< Start signal
	libvmm::mutex &m_cpu_lock; //!< CPU mutex
	volatile bool m_term; //!< Signal to terminate

public:
	/**	\brief Default constructor
	 **/
	worker(libvmm::cond &started, libvmm::mutex &cpu_lock);

	/**	\brief Virtual destructor
	 **/
	virtual ~worker();

	/**	\brief Runs the worker
	 **/
	virtual void run();

	/**	\brief Terminates the work cycle
	 **/
	void terminate();

};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_H
