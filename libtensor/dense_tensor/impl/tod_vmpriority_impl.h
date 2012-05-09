#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../tod_vmpriority.h"

namespace libtensor {


template<size_t N>
void tod_vmpriority<N>::set_priority() {

    dense_tensor_base_ctrl<N, double>(m_t).req_priority(true);
}


template<size_t N>
void tod_vmpriority<N>::unset_priority() {

    dense_tensor_base_ctrl<N, double>(m_t).req_priority(false);
}


} // namespace libtensor

