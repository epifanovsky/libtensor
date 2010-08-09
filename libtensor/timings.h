#ifndef LIBTENSOR_TIMINGS_H
#define LIBTENSOR_TIMINGS_H

#include <map>
#include <libvmm/auto_lock.h>
#include <libvmm/tls.h>
#include "timer.h"
#include "global_timings.h"

namespace libtensor {


/**	\brief Timings base class
 
 	The timings class provides timing facilities for each class which
	inherit from it.
 	
 	To obtain the timing facilities a class T has to
 	 - inherit from timings with the T as the template parameter; 
 	 - friend class timings<T>;
 	 - have the variable const char* k_clazz defined;
 	 - add start_timer and stop_timer calls around the parts of the code that 
 	   should be timed;

	\ingroup libtensor
 **/
template<typename T>
class timings {
private:
	//~ typedef std::multimap<std::string, timer> map_t; 
	typedef std::map<std::string, timer*> map_t; 
	typedef std::pair<std::string, timer*> pair_t;

private:
#ifdef LIBTENSOR_TIMINGS
	static libvmm::mutex m_lock; //!< Thread safety lock
#endif // LIBTENSOR_TIMINGS

public:
	/**	\brief Virtual destructor
	 **/
	virtual ~timings() { }

protected:	
	/**	\brief Starts the default timer 
	 **/
	void start_timer();

	/**	\brief Stops the default timer and submits its duration to
			the global timings object
	 **/
	void stop_timer();

	/**	\brief Starts a custom timer 
	 	\param name Timer name.
	 **/
	void start_timer(const std::string &name);
	
	/**	\brief Stops a custom timer and submits its duration to
			the global timings object
	   	\param name Timer name
	 **/
	void stop_timer(const std::string &name);

private:
	void make_id(std::string &id, const std::string &name);

};


#ifdef LIBTENSOR_TIMINGS
template<typename T>
libvmm::mutex timings<T>::m_lock;
#endif // LIBTENSOR_TIMINGS


template<typename T>
inline void timings<T>::start_timer() {

#ifdef LIBTENSOR_TIMINGS
	start_timer("");
#endif // LIBTENSOR_TIMINGS		
}	


template<typename T>
inline void timings<T>::start_timer(const std::string &name) {

#ifdef LIBTENSOR_TIMINGS
	libvmm::auto_lock lock(m_lock);

	std::string id;
	make_id(id, name);

	map_t &timers = libvmm::tls<map_t>::get_instance().get();
	//~ typename map_t::iterator i = timers.insert(pair_t(id, timer()));
	//~ i->second.start();
	timer *t = new timer;
	std::pair<typename map_t::iterator, bool> i =
		timers.insert(pair_t(id, t));
	if(i.second == false) {
		delete t;
		throw_exc("timings<T>", "start_timer(const std::string&)",
			"Duplicate timer.");		
	}
	t->start();
#endif // LIBTENSOR_TIMINGS
}	


template<typename T>
inline void timings<T>::stop_timer() {

#ifdef LIBTENSOR_TIMINGS
	stop_timer("");
#endif // LIBTENSOR_TIMINGS
}


template<typename T>
inline void timings<T>::stop_timer(const std::string &name) {

#ifdef LIBTENSOR_TIMINGS
	libvmm::auto_lock lock(m_lock);

	std::string id;
	make_id(id, name);

	map_t &timers = libvmm::tls<map_t>::get_instance().get();
	typename map_t::iterator i = timers.find(id);
	if(i == timers.end()) {
		throw_exc("timings<T>", "stop_timer(const std::string&)",
			"No timer with this id.");		
	}

	timer *t = i->second;
	t->stop();
	timers.erase(i);
	global_timings::get_instance().add_to_timer(id, *t);
	delete t;
#endif // LIBTENSOR_TIMINGS
}


template<typename T>
void timings<T>::make_id(std::string &id, const std::string &name) {

	id = T::k_clazz;
	if(!name.empty()) {
		id += "::";
		id += name;
	}
}


} // namespace libtensor

#endif // LIBTENSOR_TIMINGS_H
