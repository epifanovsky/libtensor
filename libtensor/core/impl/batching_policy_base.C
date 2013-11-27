#include "../batching_policy_base.h"

namespace libtensor {


batching_policy_base::batching_policy_base() : m_batchsz(0) {

}


void batching_policy_base::set_batch_size(size_t batchsz) {

    batching_policy_base::get_instance().m_batchsz = batchsz;
}


size_t batching_policy_base::get_batch_size() {

    return batching_policy_base::get_instance().m_batchsz;
}


} // namespace libtensor

