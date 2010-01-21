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
		out << "Execution of " << it->first << ": " << std::endl;
		out << "Calls: " << std::setw(8) << it->second.m_calls << ", ";
		out << it->second.m_total << std::endl;
		it++;
	}
	
	return out;
}

}