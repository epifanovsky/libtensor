#ifndef LIBUTIL_LOCAL_TIMINGS_STORE_H
#define LIBUTIL_LOCAL_TIMINGS_STORE_H

#include "timings_store.h"
#include "local_timings_store.h"

namespace libutil {


template<typename Module>
class local_timings_store : public local_timings_store_base {
public:
    /** \brief Initializes the store and registers it
     **/
    local_timings_store() {
        timings_store<Module>::get_instance().register_local(this);
    }

    /** \brief Unregisters the store and destroys it
     **/
    ~local_timings_store() {
        timings_store<Module>::get_instance().unregister_local(this);
    }

};


} // namespace libutil

#endif // LIBUTIL_LOCAL_TIMINGS_STORE_H
