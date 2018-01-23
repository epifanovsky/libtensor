#ifndef LIBVMM_RWLOCK_WINDOWS_SRW_H
#define LIBVMM_RWLOCK_WINDOWS_SRW_H

#include <utility>
#include <windows.h>

namespace libutil {

/**	\brief Read-write lock implementation for Windows

	This implementation uses Slim Read/Write locks that are available
	since Vista.

	\ingroup libutil_threads
 **/
class rwlock_windows_srw {
public:
	typedef std::pair<SWRLOCK, bool> rwlock_id_t;

public:
	static void create(rwlock_id_t &id) {

		InitializeSRWLock(&id.first);
		id.second = false;
	}

	static void destroy(rwlock_id_t &id) {

	}

	static void rdlock(rwlock_id_t &id) {

		AcquireSRWLockShared(&id.first);
		id.second = false;
	}

	static void wrlock(rwlock_id_t &id) {

		AcquireSRWLockExclusive(&id.first);
		id.second = true;
	}

	static void unlock(rwlock_id_t &id) {

		if(id.second) {
			id.second = false;
			ReleaseSRWLockExclusive(&id.first);
		} else {
			ReleaseSRWLockShared(&id.first);
		}
	}

};

} // namespace libutil

#endif // LIBVMM_RWLOCK_WINDOWS_SRW_H
