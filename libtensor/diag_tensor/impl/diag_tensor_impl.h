#ifndef LIBTENSOR_DIAG_TENSOR_IMPL_H
#define LIBTENSOR_DIAG_TENSOR_IMPL_H

#include "../diag_tensor.h"

namespace libtensor {


template<size_t N, typename T, typename Alloc>
const char *diag_tensor<N, T, Alloc>::k_clazz = "diag_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
diag_tensor<N, T, Alloc>::diag_tensor(const diag_tensor_space<N> &spc) :

    m_spc(spc) {

    std::vector<size_t> ssl;
    m_spc.get_all_subspaces(ssl);
    for(size_t i = 0; i < ssl.size(); i++) {
        std::pair<size_t, pointer_record> ppr;
        ppr.first = ssl[i];
        ppr.second.vptr = Alloc::allocate(m_spc.get_subspace_size(ssl[i]));
        m_ptr.insert(ppr);
    }
}


template<size_t N, typename T, typename Alloc>
diag_tensor<N, T, Alloc>::~diag_tensor() {

    for(typename std::map<size_t, pointer_record>::iterator i = m_ptr.begin();
        i != m_ptr.end(); ++i) {

        Alloc::deallocate(i->second.vptr);
    }
}


template<size_t N, typename T, typename Alloc>
diag_tensor<N, T, Alloc>::session_handle_type
diag_tensor<N, T, Alloc>::on_req_open_session() {

    return 0;
}


template<size_t N, typename T, typename Alloc>
void diag_tensor<N, T, Alloc>::on_req_close_session(
    const session_handle_type &h) {

}


template<size_t N, typename T, typename Alloc>
const T *diag_tensor<N, T, Alloc>::on_req_const_dataptr(
    const session_handle_type &h, size_t ssn) {

    static const char *method = "on_req_const_dataptr()";

    typename std::map<size_t, pointer_record>::iterator iptr = m_ptr.find(ssn);

    if(iptr == m_ptr.end()) {
        //  Subspace data pointer not found
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "ssn");
    }

    pointer_record &pr = iptr->second;

    if(pr.ptrcnt > 0) {
        //  Data pointer has been checked out for writing
        throw 0;
    }

    if(pr.const_ptrcnt > 0) {
        //  Data pointer has been checked out for reading
        pr.const_ptrcnt++;
        return pr.const_dataptr;
    }

    pr.const_dataptr = Alloc::lock_ro(pr.vptr);
    pr.const_ptrcnt = 1;
    return pr.const_dataptr;
}


template<size_t N, typename T, typename Alloc>
void diag_tensor<N, T, Alloc>::on_ret_const_dataptr(
    const session_handle_type &h, size_t ssn, const T *p) {

    static const char *method = "on_ret_const_dataptr()";

    typename std::map<size_t, pointer_record>::iterator iptr = m_ptr.find(ssn);

    if(iptr == m_ptr.end()) {
        //  Subspace data pointer not found
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "ssn");
    }

    pointer_record &pr = iptr->second;

    if(pr.const_ptrcnt == 0) {
        //  All pointer have been checked in
        throw 0;
    }

    if(pr.const_dataptr != p) {
        //  Bad pointer
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "p");
    }

    pr.const_ptrcnt--;
    if(pr.const_ptrcnt == 0) {
        Alloc::unlock_ro(pr.vptr);
        pr.const_dataptr = 0;
    }
}


template<size_t N, typename T, typename Alloc>
T *diag_tensor<N, T, Alloc>::on_req_dataptr(const session_handle_type &h,
    size_t ssn) {

    static const char *method = "on_req_dataptr()";

    typename std::map<size_t, pointer_record>::iterator iptr = m_ptr.find(ssn);

    if(iptr == m_ptr.end()) {
        //  Subspace data pointer not found
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "ssn");
    }

    pointer_record &pr = iptr->second;

    if(pr.ptrcnt > 0 || pr.const_ptrcnt > 0) {
        //  Data pointer has been checked out for reading or writing already
        throw 0;
    }

    pr.dataptr = Alloc::lock_rw(pr.vptr);
    pr.ptrcnt = 1;
    return pr.dataptr;
}


template<size_t N, typename T, typename Alloc>
void diag_tensor<N, T, Alloc>::on_ret_dataptr(const session_handle_type &h,
    size_t ssn, T *p) {

    static const char *method = "on_ret_dataptr()";

    typename std::map<size_t, pointer_record>::iterator iptr = m_ptr.find(ssn);

    if(iptr == m_ptr.end()) {
        //  Subspace data pointer not found
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "ssn");
    }

    pointer_record &pr = iptr->second;

    if(pr.ptrcnt == 0) {
        //  All pointer have been checked in
        throw 0;
    }

    if(pr.dataptr != p) {
        //  Bad pointer
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "p");
    }

    Alloc::unlock_rw(pr.vptr);
    pr.ptrcnt = 0;
    pr.dataptr = 0;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_IMPL_H

