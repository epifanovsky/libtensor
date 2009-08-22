#ifndef TIMER_H_
#define TIMER_H_

#include "exception.h"

#include <iostream>
#include <ctime>

#ifdef POSIX
#include <unistd.h>
#include <sys/times.h>
#endif

namespace libtensor {

class time_pt_t;
//! comparison operator of two time differences
bool operator<=( const time_pt_t&, const time_pt_t& );
//! prints time point to ostream
std::ostream& operator<<( std::ostream&, const time_pt_t& );

class time_diff_t;
//! equal comparison of time differences
bool operator==( const time_diff_t&, const time_diff_t& );
//! unequal comparison of time differences
bool operator!=( const time_diff_t&, const time_diff_t& );
//! prints time difference to ostream
std::ostream& operator<<( std::ostream&, const time_diff_t& );
//! form the time difference between two time points
time_diff_t operator-( const time_pt_t&, const time_pt_t& );

/** \brief Determine a point in time
  
	Stores the point in time when function now() is called.    
 **/
class time_pt_t {
	friend time_diff_t operator-( const time_pt_t&, const time_pt_t& );
	friend bool operator<=( const time_pt_t&, const time_pt_t& );
	friend std::ostream& operator<<( std::ostream&, const time_pt_t& );
	
	clock_t m_rt; //!< real or wall time
#ifdef POSIX
	clock_t m_ut; //!< user time 
	clock_t m_st; //!< system time
#endif 

public:
	//! saves point in time
	void now(); 
};


/** \brief Stores a time difference
 **/
class time_diff_t 
{	
	friend time_diff_t operator-( const time_pt_t&, const time_pt_t& );
	friend bool operator==( const time_diff_t&, const time_diff_t& );
	friend std::ostream& operator<<( std::ostream&, const time_diff_t& );

	double m_rt; //!< total time in s	
#ifdef POSIX
	double m_ut; //!< user time in s
	double m_st; //!< system time in s
#endif
public:
	//! default constructor  
	time_diff_t( double d=0.0 ); 
		
	//! assignment operator to some value  
	time_diff_t& operator=( double d ); 
	
	//! add time difference to this 
	time_diff_t& operator+=( const time_diff_t& t );
	
	//! subtract time difference from this 
	time_diff_t& operator-=( const time_diff_t& t );
	
	double wall_time() { return m_rt; }
	double user_time();
	double system_time();
}; 


/** \brief Simple timer class
  
 	Stores the point in time when start() is called and calculates the time 
 	difference to this point as soon as stop() is called.
 	
 	\ingroup libtensor_core
 **/
class timer {
	time_pt_t m_start; //!< start time
	time_diff_t m_elapsed; //!< elapsed time 

public:
	/** \brief start the timer
	 */ 
	void start() {	m_start.now(); m_elapsed=0.0; }	
	
	/** \brief stop the timer and save the duration 
	 */ 
	void stop()	{ 
		time_pt_t end; 
		end.now(); 
#ifdef LIBTENSOR_DEBUG
		if ( m_elapsed != time_diff_t() ) 
			throw exception("libtensor","timer","stop()",__FILE__,__LINE__,
							"","Timer not started");
#endif 
		m_elapsed=(end-m_start); 
	}
	
	/** \brief retrieve the time elapsed between start and stop signal 
	 */
	time_diff_t duration() const {	return m_elapsed; }  
};



inline void time_pt_t::now() 
{
#ifdef POSIX
	static struct tms pt;
	times(&pt);
#endif 
	m_rt=clock();
#ifdef POSIX
	m_ut=pt.tms_utime;
	m_st=pt.tms_stime;
#endif 
}

inline bool operator<=( const time_pt_t& a, const time_pt_t& b ) 
{
#ifdef POSIX
	return ((a.m_rt<=b.m_rt)&&(a.m_st<=b.m_st)&&(a.m_ut<=b.m_ut));
#else
	return (a.m_rt<=b.m_rt);
#endif
}

inline time_diff_t::time_diff_t( double d )  
#ifdef POSIX  
	: m_ut(d), m_st(d), m_rt(d)
#else
	: m_rt(d) 
#endif 
{ }

inline time_diff_t& time_diff_t::operator=( double d ) 
{
#ifdef POSIX  
	m_ut=d; 
	m_st=d;
#endif 
	m_rt=d; 
	return *this; 
}

inline time_diff_t& time_diff_t::operator+=( const time_diff_t& t ) 
{ 
#ifdef POSIX  
	m_ut+=t.m_ut; 
	m_st+=t.m_st; 
#endif 
	m_rt+=t.m_rt;	
	return *this; 
} 

inline time_diff_t& time_diff_t::operator-=( const time_diff_t& t ) 
{ 
#ifdef POSIX  
	m_ut-=t.m_ut; 
	m_st-=t.m_st; 
#endif 
	m_rt-=t.m_rt; 
	return *this; 
} 

inline double time_diff_t::user_time() 
{
#ifdef POSIX  
	return m_ut;
#else
	return m_rt; 
#endif 
}

inline double time_diff_t::system_time() 
{
#ifdef POSIX  
	return m_st;
#else
	return m_rt; 
#endif 
}


inline time_diff_t operator-( const time_pt_t& end, const time_pt_t& begin ) 
{
#ifdef POSIX
	static const double clk2sec=1./sysconf(_SC_CLK_TCK);
	static const double CLK2SEC=1./CLOCKS_PER_SEC;
#endif
	
	if ( ! (begin<=end) ) 
		throw bad_parameter("libtensor","",
			"time_diff_t operator-( const time_pt_t&, const time_pt_t& )",
			__FILE__,__LINE__,"Start time later than stop time"); 
			
	time_diff_t res; 
	res.m_rt=(end.m_rt-begin.m_rt)*CLK2SEC; 
#ifdef POSIX  
	res.m_ut=(end.m_ut-begin.m_ut)*clk2sec; 
	res.m_st=(end.m_st-begin.m_st)*clk2sec; 

//	if ( res.m_ut > res.m_rt ) {
//		std::cout << std::endl << "WARNING User time larger than wall time: ";
//		std::cout << res << "; ";
//		std::cout << "User: " << end.m_ut-begin.m_ut;
//		std::cout << " (" << end.m_ut << "," << begin.m_ut;
//		std::cout << ")";    
//		std::cout << ", Wall: " << end.m_rt-begin.m_rt;
//		std::cout << " (" << end.m_rt << "," << begin.m_rt;
//		std::cout << ")" << std::endl;    
//	}
#endif

	return res; 
}

inline bool operator==( const time_diff_t& a, const time_diff_t& b ) 
{
#ifdef POSIX
	return ((a.m_rt==b.m_rt)&&(a.m_st==b.m_st)&&(a.m_ut==b.m_ut));
#else
	return (a.m_rt==b.m_rt);
#endif
}

inline bool operator!=( const time_diff_t& a, const time_diff_t& b ) {
	return !(a==b);
} 

}

#endif // TIMER_H
