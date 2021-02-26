#ifndef LIBUTIL_TIMINGS_STORE_H
#define LIBUTIL_TIMINGS_STORE_H

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <libutil/singleton.h>
#include <libutil/threads/mutex.h>
#include "timing_record.h"
#include "local_timings_store_base.h"


namespace libutil {


/** \brief Accumulates and stores timings (base class)

    \sa timings

    \ingroup libutil_timings
 **/
class timings_store_base {
public:
    typedef std::map<std::string, timing_record> map_type;
    typedef std::pair<std::string, timing_record> pair_type;

private:
    std::vector<local_timings_store_base*> m_lts; //!< Thread-local stores
    map_type m_times; //!< Map containing all run times
    mutable mutex m_lock; //!< Mutex for thread safety

public:
    void register_local(local_timings_store_base *lts);
    void unregister_local(local_timings_store_base *lts);

public:
    /** \brief Resets all timings
     **/
    void reset();

    /** \brief Returns the number of saved timings
     */
    size_t get_ntimings() const;

    /** \brief Returns timing with given id (slow, for debugging purposes only)
     **/
    time_diff_t get_time(const std::string &id) const;

    /** \brief Prints formatted timings to an output stream
     **/
    void print(std::ostream &os);

    /** \brief Prints the timings to an output stream in the CSV format
     **/
    void print_csv(std::ostream &os, char delim = ',');

};


/** \brief Accumulates and stores all timings per module

    \sa timings

    \ingroup libutil_timings
 **/
template<typename Module>
class timings_store :
    public timings_store_base,
    public libutil::singleton< timings_store<Module> > {

    friend class libutil::singleton< timings_store<Module> >;

protected:
    /** \brief Protected constructor
     **/
    timings_store() { }

};


} // namespace libutil

#endif // LIBUTIL_TIMINGS_STORE_H
