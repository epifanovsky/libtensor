#include "tensor_list.h"

namespace libtensor {
namespace iface {


tensor_list::~tensor_list() {

    for(size_t i = 0; i < m_lst.size(); i++) delete m_lst[i];
}


size_t tensor_list::get_tensor_order(unsigned tid) const {

    if(tid >= m_lst.size()) {
        throw 0;
    }

    return m_lst[tid]->get_n();
}


const std::type_info &tensor_list::get_tensor_type(unsigned tid) const {

    if(tid >= m_lst.size()) {
        throw 0;
    }

    return m_lst[tid]->get_t();
}


} // namespace iface
} // namespace libtensor
