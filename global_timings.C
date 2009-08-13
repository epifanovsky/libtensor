#include "global_timings.h"

#include "timer.h" 
#include <iomanip>

namespace libtensor {

std::ostream&
operator<<(std::ostream& out, const libtensor::global_timings& timings) 
{
	const global_timings::map_t& times=timings.m_times;
	global_timings::map_t::const_iterator it=times.begin();
	
	while ( it != times.end() ) {
		out << "Execution of " << std::setw(30) << it->first << ": ";
		out << "Calls: " << std::setw(3) << it->second.m_calls << ", ";
		out << it->second.m_total << std::endl;
		it++;
	}
	
	return out;
}

}