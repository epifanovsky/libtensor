#include "timer.h"
#include <iomanip>

namespace libutil {

std::ostream& operator<<(std::ostream& out, const time_diff_t& t)
{
	out << "User: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_ut << " s, ";
	out << "System: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_st << " s, ";
	out << "Wall: " << std::setw(8) << std::setprecision(2);
	out << std::showpoint << std::fixed << t.m_rt << " s";
	return out;
}

std::ostream& operator<<(std::ostream& out, const time_pt_t& t)
{
	out << "User: " << std::setw(8) << t.m_ut << " ticks, ";
	out << "System: " << std::setw(8) << t.m_st << " ticks, ";
	out << "Wall: " << std::setw(8) << t.m_rt << " ticks";
	return out;
}

} // namespace libutil
