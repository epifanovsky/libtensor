#ifndef LIBTENSOR_GLOBAL_TIMINGS_H
#define LIBTENSOR_GLOBAL_TIMINGS_H

#include <libvmm/singleton.h>
#include "timer.h"
#include "exception.h"
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace libtensor {

/** \brief global timings object
 */
class global_timings : public libvmm::singleton<global_timings> {
	friend class libvmm::singleton<global_timings>;
	friend std::ostream& operator<<( std::ostream&, const global_timings& );

	struct timing_t {
		time_diff_t m_total;
		size_t m_calls;

		timing_t(time_diff_t time) : m_total(time), m_calls(1)
		{}
		timing_t& operator+=( time_diff_t time )
		{ m_calls++; m_total+=time; return *this; }
	};

	typedef std::map<const std::string, timing_t> map_t;
	typedef std::pair<const std::string, timing_t> pair_t;

	map_t m_times; //!< map containing all run times
protected:
	global_timings() {}
public:
	/** \brief adds duration of timer to timing with given id
	 */
	void add_to_timer( const std::string&, const timer& );

	/** \brief resets all timers
	 */
	void reset();

	/** \brief return timing of given id
	 */
	time_diff_t get_time( const std::string& ) const;

	/** \brief get number of saved timings
	 */
	size_t ntimings() const;
};

inline void
global_timings::add_to_timer(const std::string& id, const timer& t )
{
	map_t::iterator it = m_times.find(id);
	if ( it == m_times.end() )
		m_times.insert( pair_t(id,timing_t(t.duration())) );
	else
		it->second+=t.duration();
}

inline void
global_timings::reset()
{
	m_times.clear();
}

inline time_diff_t
global_timings::get_time( const std::string& id ) const
{
	map_t::const_iterator it = m_times.find(id);
	if ( it == m_times.end() )
		throw_exc("global_timings","get_time(const char*) const","No timer with this id");

	return it->second.m_total;
}

inline size_t
global_timings::ntimings() const
{
	return m_times.size();
}

}

#endif // LIBTENSOR_GLOBAL_TIMINGS_H
