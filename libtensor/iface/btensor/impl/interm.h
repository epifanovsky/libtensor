#ifndef LIBTENSOR_IFACE_INTERM_H
#define LIBTENSOR_IFACE_INTERM_H

#include <map>
#include <libtensor/iface/tensor_list.h>
#include "btensor_placeholder.h"

namespace libtensor {
namespace iface {


class interm : public noncopyable {
public:
    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    tensor_list &m_tl; //!< Tensor list
    std::map<tid_t, btensor_placeholder_base*> m_interm; //!< Intermediates

public:
    interm(tensor_list &tl) : m_tl(tl) {

    }

    ~interm() {
        for(std::map<tid_t, btensor_placeholder_base*>::iterator i =
            m_interm.begin(); i != m_interm.end(); ++i) {
            delete i->second;
        }
    }

    template<size_t N, typename T>
    tid_t create_interm() {

        tid_t tid;
        btensor_placeholder<N, T> *p = new btensor_placeholder<N, T>;
        try {
            tid = m_tl.get_tensor_id(*p);
            m_interm.insert(std::make_pair(tid, p));
            std::cout << "create_interm: " << (void*)tid << " " << p << std::endl;
        } catch(...) {
            delete p;
            throw;
        }
        return tid;
    }

    void destroy_interm(tid_t tid) {

        std::map<tid_t, btensor_placeholder_base*>::iterator i =
            m_interm.find(tid);
        if(i == m_interm.end()) throw 123;
        delete i->second;
        m_interm.erase(i);
    }

    bool is_interm(tid_t tid) const {
        return m_interm.count(tid) > 0;
    }

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_INTERM_H
