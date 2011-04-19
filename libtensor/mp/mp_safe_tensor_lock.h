#ifndef LIBTENSOR_MP_SAFE_TENSOR_LOCK_H
#define LIBTENSOR_MP_SAFE_TENSOR_LOCK_H

#include <libvmm/mutex.h>
#include <libvmm/singleton.h>

namespace libtensor {


/**	\brief Global lock for mp_safe_tensor objects

	\ingroup libtensor_mp
 **/
class mp_safe_tensor_lock : public libvmm::singleton<mp_safe_tensor_lock> {
	friend class libvmm::singleton<mp_safe_tensor_lock>;

private:
	libvmm::mutex m_lock; //!< Lock object

protected:
	mp_safe_tensor_lock() { }

public:
	void lock() {
		m_lock.lock();
	}

	void unlock() {
		m_lock.unlock();
	}

};


} // namespace libtensor

#endif // LIBTENSOR_MP_SAFE_TENSOR_LOCK_H
