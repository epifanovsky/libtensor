#ifndef LIBTENSOR_THREADS_H
#define LIBTENSOR_THREADS_H

#include <libvmm/auto_lock.h>
#include <libvmm/auto_spinlock.h>
#include <libvmm/cond.h>
#include <libvmm/mutex.h>
#include <libvmm/rwlock.h>
#include <libvmm/spinlock.h>
#include <libvmm/thread.h>
#include <libvmm/tls.h>

namespace libtensor {


using libvmm::auto_lock;
using libvmm::auto_spinlock;
using libvmm::cond;
using libvmm::mutex;
using libvmm::rwlock;
using libvmm::spinlock;
using libvmm::thread;
using libvmm::tls;


} // namespace libtensor

#endif // LIBTENSOR_THREADS_H
