#ifndef LIBTENSOR_BATCHING_POLICY_BASE_H
#define LIBTENSOR_BATCHING_POLICY_BASE_H

#include <cstdlib> // for size_t
#include <libutil/singleton.h>

namespace libtensor {


class batching_policy_base : public libutil::singleton<batching_policy_base> {
    friend class libutil::singleton<batching_policy_base>;

private:
    size_t m_batchsz; //!< Batch size

protected:
    batching_policy_base();

public:
    static void set_batch_size(size_t batchsz);
    static size_t get_batch_size();

};


} // namespace libtensor

#endif // LIBTENSOR_BATCHING_POLICY_BASE_H

