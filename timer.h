#ifndef TIMER_H_
#define TIMER_H_

#include <ostream>
#include <ctime>

#ifdef POSIX
#include <unistd.h>
#include <sys/times.h>
#endif

namespace libtensor {

struct times_t;
std::ostream& operator<<( std::ostream&, const times_t& ); 

/** \brief structure to store the time */
struct times_t {
	static const double clk2sec; //!< constant to convert clock_t into seconds
	
	std::clock_t m_rt; //!< total time
	
#ifdef POSIX
	std::clock_t m_ut; //!< user time
	std::clock_t m_st; //!< system time
#endif
	
	/** \brief default constructor */
	times_t( clock_t d=0 ) 
#ifdef POSIX
		: m_rt(d), m_ut(d), m_st(d)
#else  	
		: m_rt(d)
#endif
	{}	
	
	/** \brief assignment operator to a clock_t value */
	times_t& operator=( clock_t d ) 
	{
#ifdef POSIX  
		m_ut=d; 
		m_st=d;
#endif 
		m_rt=d; 
		return *this; 
	}
	
	/** \brief operator+= for times_t */
	times_t& operator+=( const times_t& t ) 
	{ 
#ifdef POSIX  
		m_ut+=t.m_ut; 
		m_st+=t.m_st; 
#endif 
		m_rt+=t.m_rt;	
		return *this; 
	} 
	
	/** \brief operator-= for times_t */
	times_t& operator-=( const times_t& t )	
	{ 
#ifdef POSIX  
		m_ut-=t.m_ut; 
		m_st-=t.m_st; 
#endif 
		m_rt-=t.m_rt; 
		return *this; 
	} 
	
}; 


/** \brief operator- for times_t */
inline times_t operator-( const times_t& a, const times_t& b ) { times_t res(a); res-=b; return res; }

/** \brief equal comparison of times_t objects */
inline bool operator==( const times_t& a, const times_t& b ) 
{
#ifdef POSIX 
	return ((a.m_ut==b.m_ut)&&(a.m_rt==b.m_rt)&&(a.m_st==b.m_st));
#else
	return (a.m_rt==b.m_rt);
#endif 
}

/** \brief unequal comparison of times_t objects */
inline bool operator!=( const times_t& a, const times_t& b ) { return (!(a==b)); }

/** \brief Simple timer class
 */
class timer {
	times_t m_start; //!< start time
	times_t m_elapsed; //!< elapsed time 

public:
	/** \brief start the timer
	 */ 
	void start() 
	{
#ifdef POSIX
		tms start;
		m_start.m_rt=times(&start);
		m_start.m_ut=start.tms_utime;
		m_start.m_st=start.tms_stime;
#else	
		m_start=clock();
#endif
		m_elapsed=0;
	}
	
	
	/** \brief stop the timer and save the duration 
	 */ 
	void stop()	
	{ 
#ifdef POSIX
		tms end;	
		m_elapsed.m_rt=times(&end)-m_start.m_rt;	
		m_elapsed.m_ut=end.tms_utime-m_start.m_ut;
		m_elapsed.m_st=end.tms_stime-m_start.m_st;
#else	
		m_elapsed=clock()-m_start;	
#endif	
	}
	
	/** \brief retrieve the time elapsed between start and stop signal 
	 */
	times_t duration() const {
		return m_elapsed;
	}  
};

}

#endif /*TIMER_H_*/
