#ifndef LIBTENSOR_TIMER_DEFS_H
#define LIBTENSOR_TIMER_DEFS_H

namespace libtensor {

class time_pt_t;
class time_diff_t;

//! comparison operator of two time points
bool operator<=(const time_pt_t& p1, const time_pt_t& p2);

//! prints time point to ostream
std::ostream& operator<<(std::ostream& out, const time_pt_t& pt);

//! equal comparison of time differences
bool operator<=(const time_diff_t& d1, const time_diff_t& d2);

//! prints time difference to ostream
std::ostream& operator<<(std::ostream& out, const time_diff_t& d);

/** \brief Determine a point in time

	Stores the point in time when function now() is called.
 **/
class time_pt_t {

	friend class time_diff_t;
	friend bool operator<=(const time_pt_t&, const time_pt_t&);
	friend std::ostream& operator<<(std::ostream&, const time_pt_t&);

private:
	clock_t m_rt; //!< real or wall time

public:
	//! saves point in time
	void now() { m_rt = clock(); }
};


/** \brief Stores a time difference
 **/
class time_diff_t {

	friend bool operator<=(const time_diff_t&, const time_diff_t&);
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

#ifdef LIBTENSOR_DEBUG
		if (! (begin <= end)) {
			throw bad_parameter("libtensor", "time_diff_t",
				"time_diff_t operator-(const time_pt_t&, const time_pt_t&)",
				__FILE__, __LINE__, "Start time later than stop time");
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

inline time_diff_t operator-(const time_pt_t& end, const time_pt_t& begin) {
	return time_diff_t(begin, end);
}

inline bool operator<=(const time_diff_t& d1, const time_diff_t& d2) {
	return (d1.m_rt <= d2.m_rt);
}

} // namespace libtensor

#endif // LIBTENSOR_TIMER_DEFS_H
