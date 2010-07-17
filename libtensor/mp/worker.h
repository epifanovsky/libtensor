#ifndef LIBTENSOR_WORKER_H
#define LIBTENSOR_WORKER_H

#include <libvmm/cond.h>
#include <libvmm/thread.h>

namespace libtensor {


/**	\brief Worker thread

	\ingroup libtensor_mp
 **/
class worker : public libvmm::thread {
private:
	libvmm::cond &m_started; //!< Start signal
	volatile bool m_term; //!< Signal to terminate

public:
	/**	\brief Default constructor
	 **/
	worker(libvmm::cond &started);

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
