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


void global_timings::print_csv(std::ostream &os) {

	map_t::const_iterator i = m_times.begin();
	for(; i != m_times.end(); i++) {

		os << i->first << "," << i->second.m_calls << ",";
		os << std::setprecision(2) << std::showpoint << std::fixed
			<< i->second.m_total.user_time() << ",";
		os << std::setprecision(2) << std::showpoint << std::fixed
			<< i->second.m_total.system_time() << ",";
		os << std::setprecision(2) << std::showpoint << std::fixed
			<< i->second.m_total.wall_time() << std::endl;
	}
}

}
