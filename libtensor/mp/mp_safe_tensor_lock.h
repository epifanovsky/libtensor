#ifndef LIBTENSOR_MP_SAFE_TENSOR_LOCK_H
#define LIBTENSOR_MP_SAFE_TENSOR_LOCK_H

#include <libutil/singleton.h>
#include "threads.h"

namespace libtensor {


/** \brief Global lock for mp_safe_tensor objects

    \ingroup libtensor_mp
 **/
class mp_safe_tensor_lock : public libutil::singleton<mp_safe_tensor_lock> {
    friend class libutil::singleton<mp_safe_tensor_lock>;

private:
    mutex m_lock; //!< Lock object

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
