#include "../dense_tensor_ctrl.h"
#include "../to_vmpriority.h"

namespace libtensor {


template<size_t N, typename T>
void to_vmpriority<N, T>::set_priority() {

    dense_tensor_base_ctrl<N, T>(m_t).req_priority(true);
}


template<size_t N, typename T>
void to_vmpriority<N, T>::unset_priority() {

    dense_tensor_base_ctrl<N, T>(m_t).req_priority(false);
}


} // namespace libtensor

