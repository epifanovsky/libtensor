#ifndef LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

#include <libtensor/exception.h>
#include "../diag_tensor_space.h"

namespace libtensor {


template<size_t N>
const char *diag_tensor_subspace<N>::k_clazz = "diag_tensor_subspace<N>";


template<size_t N>
diag_tensor_subspace<N>::diag_tensor_subspace(size_t n) :

    m_diag(n, mask<N>()) {

    static const char *method = "diag_tensor_subspace(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n > N) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

}


template<size_t N>
const mask<N> &diag_tensor_subspace<N>::get_diag_mask(size_t n) const {

    static const char *method = "get_diag_mask(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_diag.size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

    return m_diag[n];
}


template<size_t N>
void diag_tensor_subspace<N>::set_diag_mask(size_t n, const mask<N> &msk) {

    static const char *method = "set_diag_mask(size_t, const mask<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_diag.size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }

    // Check that there is no overlap with existing masks
    // except the one being replaced
    mask<N> msk0;
    for(size_t i = 0; i < m_diag.size(); i++) {
        if(i == n) continue;
        mask<N> m(msk);
        m &= m_diag[i];
        if(!m.equals(msk0)) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "msk");
        }
    }
#endif // LIBTENSOR_DEBUG

    m_diag[n] = msk;

    mask<N> totmsk;
    for(size_t i = 0; i < m_diag.size(); i++) totmsk |= m_diag[i];
    m_msk = totmsk;
}


template<size_t N>
bool diag_tensor_subspace<N>::equals(
    const diag_tensor_subspace<N> &other) const {

    std::vector<int> chk2(other.m_diag.size(), 0);

    for(size_t i1 = 0; i1 < m_diag.size(); i1++) {

        const mask<N> &m = m_diag[i1];

        // Skip trivial masks and diagonals
        size_t nset = 0;
        for(size_t i = 0; i < N; i++) if(m[i]) nset++;
        if(nset < 2) continue;

        bool found_match = false;
        for(size_t i2 = 0; i2 < chk2.size(); i2++) {
            if(m.equals(other.m_diag[i2])) {
                chk2[i2] = 1;
                found_match = true;
                break;
            }
        }
        if(!found_match) return false;
    }

    for(size_t i2 = 0; i2 < chk2.size(); i2++) {

        if(chk2[i2]) continue;

        const mask<N> &m = other.m_diag[i2];
        size_t nset = 0;
        for(size_t i = 0; i < N; i++) if(m[i]) nset++;
        if(nset >= 2) return false;
    }

    return true;
}


template<size_t N>
void diag_tensor_subspace<N>::permute(const permutation<N> &perm) {

    for(size_t i = 0; i < m_diag.size(); i++) m_diag[i].permute(perm);
    m_msk.permute(perm);
}


template<size_t N>
const char *diag_tensor_space<N>::k_clazz = "diag_tensor_space<N>";


template<size_t N>
diag_tensor_space<N>::diag_tensor_space(const dimensions<N> &dims) :

    m_dims(dims), m_nss(0) {

}


template<size_t N>
diag_tensor_space<N>::diag_tensor_space(const diag_tensor_space &other) :

    m_dims(other.m_dims), m_nss(other.m_nss) {

    for(size_t i = 0; i < other.m_ss.size(); i++) {
        diag_tensor_subspace<N> *ss = 0;
        if(other.m_ss[i]) ss = new diag_tensor_subspace<N>(*other.m_ss[i]);
        m_ss.push_back(ss);
    }
}


template<size_t N>
diag_tensor_space<N>::~diag_tensor_space() {

    for(size_t i = 0; i < m_ss.size(); i++) delete m_ss[i];
}


template<size_t N>
void diag_tensor_space<N>::get_all_subspaces(std::vector<size_t> &ss) const {

    ss.clear();
    for(size_t i = 0; i < m_ss.size(); i++) if(m_ss[i]) ss.push_back(i);
}


template<size_t N>
const diag_tensor_subspace<N> &diag_tensor_space<N>::get_subspace(
    size_t n) const {

    static const char *method = "get_subspace(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_ss.size() || m_ss[n] == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

    return *m_ss[n];
}


template<size_t N>
size_t diag_tensor_space<N>::get_subspace_size(size_t n) const {

    const diag_tensor_subspace<N> &ss = get_subspace(n);
    size_t sz = 1;

    //  Count all diagonals
    for(size_t id = 0; id < ss.get_ndiag(); id++) {
        const mask<N> &m = ss.get_diag_mask(id);
        size_t j = 0;
        while(j < N && !m[j]) j++;
        if(j == N) continue;
        sz *= m_dims[j];
    }
    //  And then count all unrestricted indexes
    const mask<N> &tm = ss.get_total_mask();
    for(size_t i = 0; i < N; i++) if(!tm[i]) sz *= m_dims[i];

    return sz;
}


template<size_t N>
size_t diag_tensor_space<N>::add_subspace(const diag_tensor_subspace<N> &ss) {

    static const char *method = "add_subspace(const diag_tensor_subspace<N>&)";

    for(size_t id = 0; id < ss.get_ndiag(); id++) {
        const mask<N> &msk = ss.get_diag_mask(id);
        size_t d = 0;
        for(size_t i = 0; i < N; i++) if(msk[i]) {
            if(d == 0) d = m_dims[i];
            else if(d != m_dims[i]) {
                throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "ss");
            }
        }
    }

    size_t n = m_ss.size();
    diag_tensor_subspace<N> *ss1 = new diag_tensor_subspace<N>(ss);
    for(size_t i = 0; i < m_ss.size(); i++) if(m_ss[i] == 0) {
        n = i;
        break;
    }
    if(n == m_ss.size()) m_ss.push_back(0);
    m_ss[n] = ss1;
    m_nss++;
    return n;
}


template<size_t N>
void diag_tensor_space<N>::remove_subspace(size_t n) {

    static const char *method = "remove_subspace(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_ss.size() || m_ss[n] == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

    delete m_ss[n];
    m_ss[n] = 0;
    m_nss--;
}


template<size_t N>
void diag_tensor_space<N>::permute(const permutation<N> &perm) {

    m_dims.permute(perm);
    for(size_t i = 0; i < m_ss.size(); i++) if(m_ss[i]) m_ss[i]->permute(perm);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

