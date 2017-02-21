#ifndef LIBUTIL_TIMER_DEFS_H
#define LIBUTIL_TIMER_DEFS_H

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
	clock_t m_rt; //!< real or wall time

public:
	time_pt_t() : m_rt(0) { }

	//! Stores current point in time
	void now() { m_rt = clock(); }
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

		static const double clk2sec = 1. / CLOCKS_PER_SEC;

#ifdef LIBUTIL_DEBUG
		if (! (begin <= end)) {
			throw timings_exception("time_diff_t",
				"time_diff_t(const time_pt_t&, const time_pt_t&)",
				__FILE__, __LINE__, "Negative time difference.");
		}
#endif

		m_rt = (end.m_rt - begin.m_rt) * clk2sec;
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
	return (p1.m_rt <= p2.m_rt);
}
inline bool operator==(const time_pt_t& p1, const time_pt_t& p2) {
	return (p1.m_rt == p2.m_rt);
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

#endif // LIBUTIL_TIMER_DEFS_H
