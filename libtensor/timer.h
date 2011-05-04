#ifndef TIMER_H_
#define TIMER_H_

#include "exception.h"

#include <iostream>
#include <ctime>

#ifdef POSIX
#include "timer_defs_posix.h"
#else
#include "timer_defs.h"
#endif

namespace libtensor {

/** \brief Simple timer class

 	Stores the point in time when start() is called and calculates the time
 	difference to this point as soon as stop() is called.

 	\ingroup libtensor_core_time
 **/
class timer {
private:
	time_pt_t m_start, m_end; //!< start and end time
	bool m_started; //!< started flag

public:
	/** \brief Default constructor
	 **/
	timer() : m_started(false) { }

	/** \brief start the timer
	 */
	void start() {
		m_start.now();
		m_started = true;
	}

	/** \brief stop the timer and save the duration
	 */
	void stop()	{
		m_end.now();
		m_started = false;

#ifdef LIBTENSOR_DEBUG
		if (! (m_start <= m_end))
			throw generic_exception("libtensor","timer","stop()",__FILE__,__LINE__,
						"Timer not started");
#endif
	}

	/** \brief retrieve the time elapsed between start and stop signal
	 */
	time_diff_t duration() const {
		return time_diff_t(m_start, m_end);
	}
};


}

#endif // TIMER_H
