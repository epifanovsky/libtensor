#include <algorithm>
#include "tensor_list.h"

namespace libtensor {
namespace iface {


tensor_list::tensor_list() {

}


tensor_list::tensor_list(const tensor_list &tl) {

    for(map_t::const_iterator i = tl.m_lst.begin(); i != tl.m_lst.end(); ++i) {
        m_lst.insert(std::make_pair(i->first, i->second->clone()));
    }
}


tensor_list::tensor_list(tensor_list &tl, int) {

    std::swap(tl.m_lst, m_lst);
}


tensor_list::~tensor_list() {

    for(map_t::iterator i = m_lst.begin(); i != m_lst.end(); ++i) {
        delete i->second;
    }
}


void tensor_list::merge(const tensor_list &tl) {

    for(map_t::const_iterator i = tl.m_lst.begin(); i != tl.m_lst.end(); ++i) {
        if(m_lst.count(i->first) == 0) {
            m_lst.insert(std::make_pair(i->first, i->second->clone()));
        }
    }
}


size_t tensor_list::get_tensor_order(tid_t tid) const {

    map_t::const_iterator i = m_lst.find(tid);
    if(i == m_lst.end()) {
        throw 0;
    }

    return i->second->get_n();
}


const std::type_info &tensor_list::get_tensor_type(tid_t tid) const {

    map_t::const_iterator i = m_lst.find(tid);
    if(i == m_lst.end()) {
        throw 0;
    }

    return i->second->get_t();
}


} // namespace iface
} // namespace libtensor
