#ifndef TIMINGS_H_
#define TIMINGS_H_

#include "timer.h"
#include "global_timings.h"

namespace libtensor {

/** \brief Timings base class
 
 	The timings class provides timing facilities for each class which inherit 
 	from it.
 	
 	To obtain the timing facilities a class has to
 	 - inherit from this class; 
 	 - friend the template with the class as the template parameter;
 	 - have the variable const char* k_clazz defined;
 	 - add start_timer and stop_timer calls around the parts of the code that 
 	   should be timed;
 	 
 */
template<typename T>
class timings {
	
#ifdef LIBTENSOR_TIMINGS
	typedef std::map<const std::string, timer> map_t; 
	typedef std::pair<const std::string, timer> pair_t;

	timer m_timer; //!< main timer object 
	map_t m_more_timers; //!< additional timers 
#endif			

protected:	
	/** \brief Start default timer 
	 */
	void start_timer();
	/** \brief Start timer 
	 	\param name special name of the timer 
	 	
	 	Starts the timer indicated by name. 
	 */
	void start_timer( const std::string& name );
	
	/** \brief Stop timer and submit duration to the global timings object
	  	
	  	Stops the default timer. The duration is submitted to the global 
	  	timings object.	  	
	 */
	void stop_timer();
	
	/** \brief Stop timer and submit duration to the global timings object
	   	\param name special name of the timer
	  	
	  	Stops the timer indicated by name. The duration is submitted to 
	  	the global timings object.	  	
	 */
	void stop_timer( const std::string& name );
};

template<typename T>
inline void timings<T>::start_timer() 
{
#ifdef LIBTENSOR_TIMINGS
	m_timer.start();
#endif		
}	

template<typename T>
inline void timings<T>::start_timer( const std::string& name ) 
{
#ifdef LIBTENSOR_TIMINGS
	map_t::iterator it=m_more_timers.find(name);
	if ( it != m_more_timers.end() ) 
		it->second.start();
	else {
		std::pair<map_t::iterator,bool> pos=m_more_timers.insert(pair_t(name,timer()));
		pos.first->second.start();
	}
#endif		
}	
		
template<typename T>
inline void timings<T>::stop_timer() 
{
	// add thread number if doing parallel calculation
#ifdef LIBTENSOR_TIMINGS
	m_timer.stop();
	global_timings::get_instance().add_to_timer(T::k_clazz,m_timer);
#endif		
}

template<typename T>
inline void timings<T>::stop_timer( const std::string& name ) 
{
	// add thread number if doing parallel calculation
#ifdef LIBTENSOR_TIMINGS
	map_t::iterator it=m_more_timers.find(name);
	if ( it != m_more_timers.end() ) { 
		it->second.stop();
		std::string id(T::k_clazz);
		id+="::";
		id+=name;
		global_timings::get_instance().add_to_timer(id,it->second);
	}
	else
		throw_exc("timings<T>","stop_timer(const char*)","No timer with this id.");		
#endif		
}
	
}

#endif /*TIMINGS_H_*/
