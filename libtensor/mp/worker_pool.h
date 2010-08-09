#ifndef LIBTENSOR_WORKER_POOL_H
#define LIBTENSOR_WORKER_POOL_H

#include <vector>
#include <libvmm/singleton.h>
#include "worker_group.h"

namespace libtensor {


/**	\brief Supports a pool of worker threads

	\ingroup libtensor_mp
 **/
class worker_pool : public libvmm::singleton<worker_pool> {
	friend class libvmm::singleton<worker_pool>;

public:
	static const char *k_clazz; //!< Class name

private:
	bool m_init; //!< The pool is active (true) or inactive (false)
	std::vector<worker_group*> m_groups; //!< Groups of worker threads

protected:
	worker_pool() : m_init(false) { }

public:
	/**	\brief Initializes the pool with a given number of groups with
			a given number of threads in each group
	 **/
	void init(unsigned ngroups, unsigned nthreads);

	/**	\brief Shuts down the pool and terminates all the threads
	 **/
	void shutdown();

};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_POOL_H
