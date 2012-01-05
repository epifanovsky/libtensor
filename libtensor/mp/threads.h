#ifndef LIBTENSOR_THREADS_H
#define LIBTENSOR_THREADS_H

#include <libutil/threads/auto_lock.h>
#include <libutil/threads/cond.h>
#include <libutil/threads/mutex.h>
#include <libutil/threads/rwlock.h>
#include <libutil/threads/spinlock.h>
#include <libutil/threads/thread.h>
#include <libutil/threads/tls.h>

namespace libtensor {


using libutil::auto_lock;
using libutil::cond;
using libutil::mutex;
using libutil::rwlock;
using libutil::spinlock;
using libutil::thread;
using libutil::tls;


} // namespace libtensor

#endif // LIBTENSOR_THREADS_H
