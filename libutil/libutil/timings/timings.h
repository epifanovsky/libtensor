#ifndef LIBUTIL_TIMINGS_H
#define LIBUTIL_TIMINGS_H

#include <map>
#include <libutil/threads/auto_lock.h>
#include <libutil/threads/tls.h>
#include "timer.h"
#include "local_timings_store.h"

namespace libutil {


/** \brief Timings base class
    \tparam T Timed class.
    \tparam Module Timed module tag.
    \tparam Enabled Enable/disable timers.

    The timings class provides timing facilities for each class which inherits
    from it. This template has two specializations: a full set of timing
    routines when timers are enabled, and a set of dummy timing routines
    when timers are disabled.

    To obtain the timing facilities a class T has to
     - inherit from timings with the T as the template parameter;
     - friend class timings<T>;
     - have the variable const char* k_clazz defined;
     - add start_timer and stop_timer calls around the parts of the code that
       should be timed.

    \ingroup libutil_timings
 **/
template<typename T, typename Module, bool Enabled>
class timings;


template<typename T, typename Module>
class timings<T, Module, true> {
public:
    /** \brief Virtual destructor
     **/
    virtual ~timings() { }

protected:	
    /** \brief Starts the default timer
     **/
    static void start_timer();

    /** \brief Stops the default timer and submits its duration to
            the global timings object
     **/
    static void stop_timer();

    /** \brief Starts a custom timer
        \param name Timer name.
     **/
    static void start_timer(const char *name);
	
    /** \brief Stops a custom timer and submits its duration to
            the global timings object
        \param name Timer name
     **/
    static void stop_timer(const char *name);

private:
    static void make_id(std::string &id, const std::string &name);

};


template<typename T, typename Module>
class timings<T, Module, false> {
public:
    /** \brief Virtual destructor
     **/
    virtual ~timings() { }

protected:
    /** \brief Starts the default timer
     **/
    static void start_timer() { }

    /** \brief Stops the default timer and submits its duration to
            the global timings object
     **/
    static void stop_timer() { }

    /** \brief Starts a custom timer
        \param name Timer name.
     **/
    static void start_timer(const char *name) { }

    /** \brief Stops a custom timer and submits its duration to
            the global timings object
        \param name Timer name
     **/
    static void stop_timer(const char *name) { }

};


template<typename T, typename Module>
void timings<T, Module, true>::start_timer() {

    start_timer("");
}	


template<typename T, typename Module>
void timings<T, Module, true>::start_timer(const char *name) {

    std::string id;
    make_id(id, name);

    tls< local_timings_store<Module> >::get_instance().get().start_timer(id);
}	


template<typename T, typename Module>
void timings<T, Module, true>::stop_timer() {

    stop_timer("");
}


template<typename T, typename Module>
void timings<T, Module, true>::stop_timer(const char *name) {

    std::string id;
    make_id(id, name);

    tls< local_timings_store<Module> >::get_instance().get().stop_timer(id);
}


template<typename T, typename Module>
void timings<T, Module, true>::make_id(std::string &id,
    const std::string &name) {

    id = T::k_clazz;
    if(!name.empty()) {
        id += "::";
        id += name;
    }
}


} // namespace libutil

#endif // LIBUTIL_TIMINGS_H
