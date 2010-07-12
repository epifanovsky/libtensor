#ifndef LIBTENSOR_DEFAULT_SYNC_POLICY_H
#define LIBTENSOR_DEFAULT_SYNC_POLICY_H

#include <libvmm/mutex.h>

namespace libtensor {


/**	\brief Default synchronization policy

	\ingroup libtensor_mp
 **/
class default_sync_policy {
public:
	typedef libvmm::mutex mutex_t; //!< Mutex type

};


} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYNC_POLICY_H
