#ifndef LIBTENSOR_TIMER_DEFS_POSIX_H
#define LIBTENSOR_TIMER_DEFS_POSIX_H

#include <ctime>
#include <iostream>
#include <sys/times.h>
#include <unistd.h>
#include "exception.h"

namespace libtensor {

class time_pt_t;
class time_diff_t;

//! Smaller equal comparison operator of two time points
bool operator<=(const time_pt_t& p1, const time_pt_t& p2);

//! Equal comparison operator of two time points
bool operator==(const time_pt_t& p1, const time_pt_t& p2);

//! Prints time point to ostream
std::ostream& operator<<(std::ostream& out, const time_pt_t& pt);

//! Smaller equal comparison of time differences
inline bool operator<=(const time_diff_t& d1, const time_diff_t& d2);

//! Equal comparison of time differences
inline bool operator==(const time_diff_t& d1, const time_diff_t& d2);

//! Prints time difference to ostream
std::ostream& operator<<(std::ostream& out, const time_diff_t& d);


/** \brief Determine a point in time

	Stores the point in time when function now() is called.
 **/
class time_pt_t {

	friend class time_diff_t;
	friend bool operator<=(const time_pt_t&, const time_pt_t&);
	friend bool operator==(const time_pt_t&, const time_pt_t&);
	friend std::ostream& operator<<(std::ostream&, const time_pt_t&);

private:
	clock_t m_rt; //!< wall time
	clock_t m_ut; //!< user time
	clock_t m_st; //!< system time

public:
	//! saves point in time
	void now() {
		static struct tms pt;
		m_rt = times(&pt);
		m_ut = pt.tms_utime;
		m_st = pt.tms_stime;
	}
};


/** \brief Stores a time difference
 **/
class time_diff_t
{
	friend bool operator<=(const time_diff_t&, const time_diff_t&);
	friend bool operator==(const time_diff_t&, const time_diff_t&);
	friend std::ostream& operator<<(std::ostream&, const time_diff_t&);

private:
	double m_rt; //!< total time in s
	double m_ut; //!< user time in s
	double m_st; //!< system time in s

public:
	//! Default constructor
	time_diff_t(double d = 0.0) :
		m_rt(d), m_ut(d), m_st(d) { }

	//! Constructor taking a start and an end point in time
	time_diff_t(const time_pt_t &begin, const time_pt_t &end) :
		m_rt(0.0), m_ut(0.0), m_st(0.0) {

		static const double clk2sec = 1. / sysconf(_SC_CLK_TCK);

#ifdef LIBTENSOR_DEBUG
		if (! (begin <= end)) {
			throw bad_parameter("libtensor", "time_diff_t",
					"time_diff_t operator-(const time_pt_t&, const time_pt_t&)",
					__FILE__, __LINE__, "Start time later than stop time");
		}
#endif

		m_rt = (end.m_rt - begin.m_rt) * clk2sec;
		m_ut = (end.m_ut - begin.m_ut) * clk2sec;
		m_st = (end.m_st - begin.m_st) * clk2sec;
	}

	time_diff_t& operator=(double d) {
		m_ut=d; m_st=d; m_rt=d;
		return *this;
	}


	//! Add time difference to this
	time_diff_t& operator+=(const time_diff_t& t) {
		m_rt += t.m_rt; m_ut += t.m_ut; m_st += t.m_st;
		return *this;
	}

	//! Subtract time difference from this
	time_diff_t& operator-=(const time_diff_t& t) {
		m_rt -= t.m_rt; m_ut -= t.m_ut; m_st -= t.m_st;
		return *this;
	}

	double wall_time() const { return m_rt; }
	double user_time() const { return m_ut; }
	double system_time() const { return m_st; }
};

inline bool operator<=(const time_pt_t& a, const time_pt_t& b) {
	return ((a.m_rt <= b.m_rt) && (a.m_st <= b.m_st) && (a.m_ut <= b.m_ut));
}

inline bool operator==(const time_pt_t& a, const time_pt_t& b) {
	return ((a.m_rt == b.m_rt) && (a.m_st == b.m_st) && (a.m_ut == b.m_ut));
}

inline bool operator!=(const time_pt_t& a, const time_pt_t& b) {
	return (! (a == b));
}

inline time_diff_t operator-(const time_pt_t& end, const time_pt_t& begin) {
	return time_diff_t(begin, end);
}

inline bool operator<=( const time_diff_t& a, const time_diff_t& b) {
	return ((a.m_rt <= b.m_rt) && (a.m_st <= b.m_st) && (a.m_ut <= b.m_ut));
}

inline bool operator==( const time_diff_t& a, const time_diff_t& b) {
	return ((a.m_rt == b.m_rt) && (a.m_st == b.m_st) && (a.m_ut == b.m_ut));
}

inline bool operator!=( const time_diff_t& a, const time_diff_t& b) {
	return (! (a == b));
}

}

#endif // TIMER_H
