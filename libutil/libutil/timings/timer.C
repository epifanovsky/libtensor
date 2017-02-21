#include "timer.h"
#include <iomanip>

namespace libutil {
	

std::ostream& operator<<(std::ostream& out, const time_diff_t& t)
{
#ifdef HAVE_GETTIMEOFDAY
    out << "Wall: " << std::setw(8) << std::setprecision(2);
    out << std::showpoint << std::fixed << t.m_rt << " s";
#else
#ifdef POSIX
	out << "User: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_ut << " s, ";
	out << "System: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_st << " s, ";
#endif
	out << "Wall: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_rt << " s";
#endif
	return out;
}

std::ostream& operator<<(std::ostream& out, const time_pt_t& t)
{
#ifdef HAVE_GETTIMEOFDAY
    out << "Wall: " << std::setw(8) << t.m_t.tv_sec << " sec, "
        << std::setw(8) << t.m_t.tv_usec << " usec";
#else
#ifdef POSIX
	out << "User: " << std::setw(8) << t.m_ut << " ticks, ";
	out << "System: " << std::setw(8) << t.m_st << " ticks, ";
#endif
	out << "Wall: " << std::setw(8) << t.m_rt << " ticks";
#endif
	return out;
}

} // namespace libutil
