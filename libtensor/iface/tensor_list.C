#include <algorithm>
#include "tensor_list.h"

namespace libtensor {
namespace iface {


tensor_list::tensor_list() {

}


tensor_list::tensor_list(const tensor_list &tl) {

    for(map_t::const_iterator it = tl.m_lst.begin();
            it != tl.m_lst.end(); it++) {
        m_lst.insert(map_t::value_type(it->first, it->second->clone()));
    }
}


tensor_list::tensor_list(tensor_list &tl, int) {

    std::swap(tl.m_lst, m_lst);
}


tensor_list::~tensor_list() {

    for(size_t i = 0; i < m_lst.size(); i++) delete m_lst[i];
}


void tensor_list::merge(const tensor_list &tl) {

    for (map_t::const_iterator it = tl.m_lst.begin();
            it != tl.m_lst.end(); it++) {

        if (m_lst.count(it->first) == 0) {
            m_lst.insert(map_t::value_type(it->first, it->second->clone()));
        }
    }
}


size_t tensor_list::get_tensor_order(size_t tid) const {

    map_t::const_iterator it = m_lst.find(tid);
    if(it == m_lst.end()) {
        throw 0;
    }

    return it->second->get_n();
}


const std::type_info &tensor_list::get_tensor_type(size_t tid) const {

    map_t::const_iterator it = m_lst.find(tid);
    if(it == m_lst.end()) {
        throw 0;
    }

    return it->second->get_t();
}


} // namespace iface
} // namespace libtensor
