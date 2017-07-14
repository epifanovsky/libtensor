#ifndef LIBUTIL_TIMER_DEFS_GETTIMEOFDAY_H
#define LIBUTIL_TIMER_DEFS_GETTIMEOFDAY_H

#include <sys/time.h>
#include <libutil/exceptions/util_exceptions.h>

namespace libutil {

class time_pt_t;
class time_diff_t;

/** \brief Smaller equal comparison operator of two time points
	\ingroup libtensor_core_time
 **/
bool operator<=(const time_pt_t& p1, const time_pt_t& p2);

/** \brief Equal comparison operator of two time points
	\ingroup libtensor_core_time
 **/
bool operator==(const time_pt_t& p1, const time_pt_t& p2);

/** \brief Prints time point to output stream
	\ingroup libtensor_core_time
 **/
std::ostream& operator<<(std::ostream& out, const time_pt_t& pt);

/** \brief Smaller equal comparison of time differences
	\ingroup libtensor_core_time
 **/
bool operator<=(const time_diff_t& d1, const time_diff_t& d2);

/** \brief Equal comparison of time differences
	\ingroup libtensor_core_time
 **/
bool operator==(const time_diff_t& d1, const time_diff_t& d2);

/** \brief Prints time difference to output stream
	\ingroup libtensor_core_time
 **/
std::ostream& operator<<(std::ostream& out, const time_diff_t& d);

/** \brief Determine a point in time

	Stores the point in time when function now() is called.

	\ingroup libtensor_core_time
 **/
class time_pt_t {

	friend class time_diff_t;
	friend bool operator<=(const time_pt_t&, const time_pt_t&);
	friend bool operator==(const time_pt_t&, const time_pt_t&);
	friend std::ostream& operator<<(std::ostream&, const time_pt_t&);

private:
	struct timeval m_t; //!< Time

public:
	time_pt_t() { m_t.tv_sec = 0; m_t.tv_usec = 0; }

	//! Stores current point in time
	void now() { gettimeofday(&m_t, 0); }
};


/** \brief Stores a time difference

	\ingroup libtensor_core_time
 **/
class time_diff_t {

	friend bool operator<=(const time_diff_t&, const time_diff_t&);
	friend bool operator==(const time_diff_t&, const time_diff_t&);
	friend std::ostream& operator<<(std::ostream&, const time_diff_t&);

private:
	double m_rt; //!< total time in s

public:
	//! Default constructor
	time_diff_t(double d = 0.0) : m_rt(d) { }

	//! Constructor taking a start and an end point in time
	time_diff_t(const time_pt_t &begin, const time_pt_t &end) :
		m_rt(0.0) {

#ifdef LIBUTIL_DEBUG
		if (! (begin <= end)) {
			throw timings_exception("time_diff_t",
				"time_diff_t(const time_pt_t&, const time_pt_t&)",
				__FILE__, __LINE__, "Negative time difference.");
		}
#endif

		m_rt = double(end.m_t.tv_sec - begin.m_t.tv_sec) +
		    double(1000000 + end.m_t.tv_usec - begin.m_t.tv_usec) * 1e-6 - 1.0;
	}

	time_diff_t& operator=(double d) {

		m_rt = d;
		return *this;
	}

	//! Add time difference to this
	time_diff_t& operator+=(const time_diff_t& t) {

		m_rt += t.m_rt;
		return *this;
	}

	//! Subtract time difference from this
	time_diff_t& operator-=(const time_diff_t& t) {

		m_rt -= t.m_rt;
		return *this;
	}

	double wall_time() const { return m_rt; }
	double user_time() const { return m_rt; }
	double system_time() const { return m_rt; }
};


inline bool operator<=(const time_pt_t& p1, const time_pt_t& p2) {
    if(p1.m_t.tv_sec == p2.m_t.tv_sec) {
        return p1.m_t.tv_usec <= p2.m_t.tv_usec;
    } else {
        return p1.m_t.tv_sec < p2.m_t.tv_sec;
    }
}
inline bool operator==(const time_pt_t& p1, const time_pt_t& p2) {
	return (p1.m_t.tv_sec == p2.m_t.tv_sec && p1.m_t.tv_usec == p2.m_t.tv_usec);
}

inline bool operator!=(const time_pt_t& a, const time_pt_t& b) {
        return (! (a == b));
}

inline time_diff_t operator-(const time_pt_t& end, const time_pt_t& begin) {
	return time_diff_t(begin, end);
}

inline bool operator<=(const time_diff_t& d1, const time_diff_t& d2) {
	return (d1.m_rt <= d2.m_rt);
}

inline bool operator==(const time_diff_t& d1, const time_diff_t& d2) {
	return (d1.m_rt == d2.m_rt);
}

inline bool operator!=( const time_diff_t& a, const time_diff_t& b) {
        return (! (a == b));
}

} // namespace libutil

#endif // LIBUTIL_TIMER_DEFS_GETTIMEOFDAY_H
