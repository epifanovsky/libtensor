#include "timer.h"
#include <iomanip>

namespace libtensor {
	
#ifdef POSIX
const double times_t::clk2sec=1./sysconf(_SC_CLK_TCK);
#else
const double times_t::clk2sec=1./CLOCKS_PER_SEC;
#endif

std::ostream& operator<<( std::ostream& out, const times_t& t ) 
{
#ifdef POSIX
	out << "User: " << std::setw(8) << std::setprecision(2) << times_t::clk2sec*t.m_ut << " s, ";
	out << "System: " << std::setw(8) << std::setprecision(2) << times_t::clk2sec*t.m_st << " s, ";
#endif
	out << "Wall: " << std::setw(8) << std::setprecision(2) << times_t::clk2sec*t.m_rt << " s";
	return out;
}

}