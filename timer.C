#include "timer.h"
#include <iomanip>

namespace libtensor {
	

std::ostream& operator<<( std::ostream& out, const time_diff_t& t ) 
{
#ifdef POSIX
	out << "User: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_ut << " s, ";
	out << "System: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_st << " s, ";
#endif
	out << "Wall: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_rt << " s";
	return out;
}

}