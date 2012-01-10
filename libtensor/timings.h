#ifndef LIBTENSOR_TIMINGS_H
#define LIBTENSOR_TIMINGS_H

#include <libutil/timings/timings.h>

namespace libtensor {


/** \brief Timings base class
 
    The timings class provides timing facilities for each class which
    inherit from it.
 	
    To obtain the timing facilities a class T has to
     - inherit from timings with the T as the template parameter;
     - friend class timings<T>;
     - have the variable const char* k_clazz defined;
     - add start_timer and stop_timer calls around the parts of the code that
       should be timed;

    \ingroup libtensor_core
 **/
template<typename T> class timings;


#ifdef LIBTENSOR_TIMINGS

template<typename T>
class timings : public libutil::timings<T, true> { };

#else

template<typename T>
class timings : public libutil::timings<T, false> { };

#endif // LIBTENSOR_TIMINGS


} // namespace libtensor

#endif // LIBTENSOR_TIMINGS_H
