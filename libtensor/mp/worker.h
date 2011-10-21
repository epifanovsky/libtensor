#ifndef LIBTENSOR_WORKER_H
#define LIBTENSOR_WORKER_H

#include "../timings.h"
#include "cpu_pool.h"
#include "threads.h"

namespace libtensor {


/** \brief Worker thread

    \ingroup libtensor_mp
 **/
class worker : public thread, public timings<worker> {
public:
    static const char *k_clazz; //!< Class name

private:
    cond &m_started; //!< Start signal
    cpu_pool &m_cpus; //!< Pool of CPUs
    volatile bool m_term; //!< Signal to terminate

public:
    /**	\brief Initializes the worker thread
     **/
    worker(cond &started, cpu_pool &cpus);

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
