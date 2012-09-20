#ifndef LIBTENSOR_LINALG_TIMINGS_H
#define LIBTENSOR_LINALG_TIMINGS_H

#include <libutil/timings/timings.h>

namespace libtensor {


/** \brief Linear algebra timings base class
 
    The timings class provides timing facilities for each class which
    inherit from it.
     
    To obtain the timing facilities a class T has to
     - inherit from timings with the T as the template parameter;
     - friend class timings<T>;
     - have the variable const char* k_clazz defined;
     - add start_timer and stop_timer calls around the parts of the code that
       should be timed;

    \sa timings

    \ingroup libtensor_linalg
 **/
template<typename T> class linalg_timings;


/** \brief Tag for all linear algebra timings in libtensor

    \ingroup libtensor_linalg
 **/
struct libtensor_linalg_timings { };


#ifdef LIBTENSOR_LINALG_TIMINGS

template<typename T>
class linalg_timings :
    public libutil::timings<T, libtensor_linalg_timings, true> { };

#else

template<typename T>
class linalg_timings :
    public libutil::timings<T, libtensor_linalg_timings, false> { };

#endif // LIBTENSOR_LINALG_TIMINGS


} // namespace libtensor

#endif // LIBTENSOR_LINALG_TIMINGS_H
