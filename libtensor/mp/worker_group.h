#ifndef LIBTENSOR_WORKER_GROUP_H
#define LIBTENSOR_WORKER_GROUP_H

#include <vector>
#include "threads.h"

namespace libtensor {

class worker;

/**	\brief Group of worker threads

	A group of worker threads share a CPU and only one of the threads
	is executed at the same time.

	\ingroup libtensor_mp
 **/
class worker_group {
private:
	std::vector<worker*> m_workers; //!< Worker threads
	std::vector<cond*> m_started; //!< Worker start signals
	mutex m_cpu_lock; //!< CPU mutex

public:
	worker_group(unsigned nthreads, cond &started);

	void start();

	void terminate();

	void join();

};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_GROUP_H
