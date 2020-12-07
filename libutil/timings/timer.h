#ifndef LIBUTIL_TIMER_H
#define LIBUTIL_TIMER_H

#include "timer_defs_posix.h"
#include <ctime>
#include <iostream>
#include <libutil/exceptions/util_exceptions.h>

namespace libutil {

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
		m_end = m_start;
		m_started = true;
	}

	/** \brief stop the timer and save the duration
	 */
	void stop()	{
		m_end.now();
		m_started = false;

#ifdef LIBUTIL_DEBUG
		if (! (m_start <= m_end))
			throw timings_exception("timer", "stop()",
			        __FILE__, __LINE__, "Timer not started");
#endif
	}

	/** \brief retrieve the time elapsed between start and stop signal
	 */
	time_diff_t duration() const {
		return time_diff_t(m_start, m_end);
	}
};


} // namespace libutil

#endif // LIBUTIL_TIMER_H
