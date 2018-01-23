#ifndef LIBUTIL_LOCAL_TIMINGS_STORE_BASE_H
#define LIBUTIL_LOCAL_TIMINGS_STORE_BASE_H

#include <map>
#include <string>
#include <vector>
#include "timing_record.h"

namespace libutil {


class local_timings_store_base {
private:
    typedef std::multimap<std::string, timer*> incomplete_map_type;
    typedef std::pair<std::string, timer*> incomplete_pair_type;
    typedef std::map<std::string, timing_record> complete_map_type;
    typedef std::pair<std::string, timing_record> complete_pair_type;

private:
    std::vector<timer*> m_timers;
    incomplete_map_type m_incomplete;
    complete_map_type m_complete;

public:
    /** \brief Initializes the store
     **/
    local_timings_store_base();

    /** \brief Cleans up and destroys the store
     **/
    ~local_timings_store_base();

    /** \brief Starts a named timer
        \param name Timer name.
     **/
    void start_timer(const std::string &name);

    /** \brief Stops a named timer and saves it
        \param name Timer name.
     **/
    void stop_timer(const std::string &name);

    /** \brief Returns true if the container is empty, false otherwise
     **/
    bool is_empty() const;

    /** \brief Merges this store's timings into the given map
     **/
    void merge(std::map<std::string, timing_record> &t);

    /** \brief Clears all timers
     **/
    void reset();

};


} // namespace libutil

#endif // LIBUTIL_LOCAL_TIMINGS_STORE_BASE_H
