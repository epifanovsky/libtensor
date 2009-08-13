#ifndef GLOBAL_TIMINGS_H_
#define GLOBAL_TIMINGS_H_

#include "../libvmm/singleton.h"
#include "timer.h"
#include "exception.h"
#include <ostream>
#include <map>
#include <string>
#include <utility>

namespace libtensor {
	
/** \brief global timings object
 */
class global_timings : public libvmm::singleton<global_timings> {
	friend libvmm::singleton<global_timings>;
	friend std::ostream& operator<<( std::ostream&, const global_timings& );

	struct timing_t {
		times_t m_total;
		size_t m_calls;
		
		timing_t(times_t time) : m_total(time), m_calls(1) 
		{}
		timing_t& operator+=( times_t time ) 
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
	times_t get_time( const std::string& ) const;
			
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

inline times_t 
global_timings::get_time( const std::string& id ) const
{
	map_t::const_iterator it = m_times.find(id);
	if ( it == m_times.end() ) 
		throw_exc("global_timings","get_time(const char*) const","No timer with this id");
	
	return it->second.m_total;
}

}

#endif /*GLOBAL_TIMINGS_H_*/
