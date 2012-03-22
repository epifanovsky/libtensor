#ifndef LIBTENSOR_DENSE_TENSOR_IMPL_H
#define LIBTENSOR_DENSE_TENSOR_IMPL_H

#include <sstream>
#include <libtensor/exception.h>
#include "../dense_tensor.h"

namespace libtensor {


template<size_t N, typename T, typename Alloc>
const char *dense_tensor<N,T,Alloc>::k_clazz = "dense_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
dense_tensor<N, T, Alloc>::dense_tensor(const dimensions<N> &dims) :

    m_dims(dims), m_data(Alloc::invalid_pointer), m_dataptr(0),
    m_const_dataptr(0), m_ptrcount(0), m_sessions(8, 0),
    m_session_ptrcount(8, 0) {

    static const char *method = "dense_tensor(const dimensions<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(m_dims.get_size() == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dims");
    }
#endif // LIBTENSOR_DEBUG

    m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
dense_tensor<N, T, Alloc>::dense_tensor(const dense_tensor_i<N, T> &t) :

    m_dims(t.get_dims()), m_data(Alloc::invalid_pointer), m_dataptr(0),
    m_const_dataptr(0), m_ptrcount(0), m_sessions(8, 0),
    m_session_ptrcount(8, 0) {

    m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
dense_tensor<N, T, Alloc>::dense_tensor(const dense_tensor<N, T, Alloc> &t) :

    m_dims(t.m_dims), m_data(Alloc::invalid_pointer), m_dataptr(0),
    m_const_dataptr(0), m_ptrcount(0), m_sessions(8, 0),
    m_session_ptrcount(8, 0) {

    m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
dense_tensor<N, T, Alloc>::~dense_tensor() {

    if(m_const_dataptr != 0) {
        Alloc::unlock_ro(m_data);
        m_const_dataptr = 0;
    } else if(m_dataptr != 0) {
        Alloc::unlock_rw(m_data);
        m_dataptr = 0;
    }
    Alloc::deallocate(m_data);
}


template<size_t N, typename T, typename Alloc>
const dimensions<N> &dense_tensor<N, T, Alloc>::get_dims() const {

    return m_dims;
}


template<size_t N, typename T, typename Alloc>
typename dense_tensor<N, T, Alloc>::handle_t
dense_tensor<N, T, Alloc>::on_req_open_session() {

    size_t sz = m_sessions.size();

    for(register size_t i = 0; i < sz; i++) {
        if(m_sessions[i] == 0) {
            m_sessions[i] = 1;
            m_session_ptrcount[i] = 0;
            return i;
        }
    }

    m_sessions.resize(2 * sz, 0);
    m_session_ptrcount.resize(2 * sz, 0);
    m_sessions[sz] = 1;
    m_session_ptrcount[sz] = 0;
    return sz;
}


template<size_t N, typename T, typename Alloc>
void dense_tensor<N, T, Alloc>::on_req_close_session(const handle_t &h) {

    verify_session(h);

    m_sessions[h] = 0;
    if(m_const_dataptr != 0) {
        m_ptrcount -= m_session_ptrcount[h];
        m_session_ptrcount[h] = 0;
        if(m_ptrcount == 0) {
            Alloc::unlock_ro(m_data);
            m_const_dataptr = 0;
        }
    } else if(m_dataptr != 0) {
        m_ptrcount = 0;
        m_session_ptrcount[h] = 0;
        Alloc::unlock_rw(m_data);
        m_dataptr = 0;
    }
}


template<size_t N, typename T, typename Alloc>
void dense_tensor<N, T, Alloc>::on_req_prefetch(const handle_t &h) {

    verify_session(h);

    if(m_dataptr == 0 && m_const_dataptr == 0) Alloc::prefetch(m_data);
}


template<size_t N, typename T, typename Alloc>
void dense_tensor<N, T, Alloc>::on_req_priority(const handle_t &h, bool pri) {

    verify_session(h);

    if(pri) Alloc::set_priority(m_data);
    else Alloc::unset_priority(m_data);
}


template<size_t N, typename T, typename Alloc>
T *dense_tensor<N, T, Alloc>::on_req_dataptr(const handle_t &h) {

    static const char *method = "on_req_dataptr(const handle_t&)";

    verify_session(h);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__, "");
    }

    if(m_dataptr != 0) {
        throw_exc(k_clazz, method,
            "Data pointer is already checked out for rw");
    }
    if(m_const_dataptr != 0) {
        throw_exc(k_clazz, method,
            "Data pointer is already checked out for ro");
    }

    timings< dense_tensor<N, T, Alloc> >::start_timer("lock_rw");
    m_dataptr = Alloc::lock_rw(m_data);
    timings< dense_tensor<N, T, Alloc> >::stop_timer("lock_rw");
    m_session_ptrcount[h] = 1;
    m_ptrcount = 1;
    return m_dataptr;
}


template<size_t N, typename T, typename Alloc>
void dense_tensor<N, T, Alloc>::on_ret_dataptr(const handle_t &h, const T *p) {

    static const char *method = "on_ret_dataptr(const handle_t&, const T*)";

    verify_session(h);

    if(m_dataptr == 0 || m_dataptr != p) {
        std::ostringstream ss;
        ss << "p[m_dataptr=" << m_dataptr << ",p=" << p << ",m_ptrcount="
            << m_ptrcount << "]";
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            ss.str().c_str());
    }

    m_session_ptrcount[h] = 0;
    m_ptrcount = 0;
    Alloc::unlock_rw(m_data);
    m_dataptr = 0;
}


template<size_t N, typename T, typename Alloc>
const T *dense_tensor<N, T, Alloc>::on_req_const_dataptr(const handle_t &h) {

    static const char *method = "on_req_const_dataptr(const handle_t&)";

    verify_session(h);

    if(m_dataptr != 0) {
        throw_exc(k_clazz, method,
            "Data pointer is already checked out for rw");
    }
    if(m_const_dataptr != 0) {
        m_session_ptrcount[h]++;
        m_ptrcount++;
        return m_const_dataptr;
    }

    timings< dense_tensor<N, T, Alloc> >::start_timer("lock_ro");
    m_const_dataptr = Alloc::lock_ro(m_data);
    timings< dense_tensor<N, T, Alloc> >::stop_timer("lock_ro");
    m_session_ptrcount[h] = 1;
    m_ptrcount = 1;
    return m_const_dataptr;
}


template<size_t N, typename T, typename Alloc>
void dense_tensor<N, T, Alloc>::on_ret_const_dataptr(const handle_t &h,
    const T *p) {

    static const char *method =
        "on_ret_const_dataptr(const handle_t&, const T*)";

    verify_session(h);

    if(m_const_dataptr == 0 || m_const_dataptr != p) {
        std::ostringstream ss;
        ss << "p[m_const_dataptr=" << m_const_dataptr << ",p=" << p
            << ",m_ptrcount=" << m_ptrcount << "]";
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            ss.str().c_str());
    }

    if(m_session_ptrcount[h] > 0) {
        m_session_ptrcount[h]--;
        m_ptrcount--;
    }
    if(m_ptrcount == 0) {
        Alloc::unlock_ro(m_data);
        m_const_dataptr = 0;
    }
}


template<size_t N, typename T, typename Alloc>
inline void dense_tensor<N, T, Alloc>::verify_session(size_t h) {

    static const char *method = "verify_session(size_t)";

    if(h >= m_sessions.size() || m_sessions[h] == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "h");
    }
}


} // namespace libtensor

#endif // LIBTENSOR_DENSE_TENSOR_IMPL_H
