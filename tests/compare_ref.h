#ifndef LIBTENSOR_COMPARE_REF_H
#define LIBTENSOR_COMPARE_REF_H

#include <cmath>
#include <sstream>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/dense_tensor/to_compare.h>
#include <libtensor/block_tensor/btod_compare.h>
#include <libtensor/exception.h>

namespace libtensor {


template<size_t N, typename T>
class compare_ref_x {
public:
    static void compare(const char *test, T d, T d_ref);

    static void compare(const char *test, dense_tensor_i<N, T> &t,
        dense_tensor_i<N, T> &t_ref, T thresh);

    static void compare(const char *test, block_tensor_rd_i<N, T> &t,
        block_tensor_rd_i<N, T> &t_ref, T thresh);

    static void compare(const char *test, const symmetry<N, T> &s,
        const symmetry<N, T> &s_ref);

    static void compare(const char *test, const block_index_space<N> &bis,
        const symmetry_element_set<N, T> &s,
        const symmetry_element_set<N, T> &s_ref);

};


template<size_t N, typename T>
void compare_ref_x<N, T>::compare(const char *test, T d, T d_ref) {

    T k_thresh;
    if(typeid(T) == typeid(double)) k_thresh = 1e-14;
    else if(typeid(T) == typeid(float)) k_thresh = 1e-5;
    if(fabs(d - d_ref) > fabs(d_ref * k_thresh)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (res), "
            << d_ref << " (ref), " << d - d_ref << " (diff)";
        throw_exc("compare_ref_x", "compare()", ss.str().c_str());
    }
}


template<size_t N, typename T>
void compare_ref_x<N, T>::compare(const char *test, dense_tensor_i<N, T> &t,
    dense_tensor_i<N, T> &t_ref, T thresh) {

    to_compare<N, T> cmp(t, t_ref, thresh);
    if(!cmp.compare()) {
        std::ostringstream ss1, ss2;
        ss2 << "In " << test << ": ";
        ss2 << "Result does not match reference at element "
            << cmp.get_diff_index() << ": "
            << cmp.get_diff_elem_1() << " (act) vs. "
            << cmp.get_diff_elem_2() << " (ref), "
            << cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
            << " (diff)";
        throw_exc("compare_ref_x", "compare()", ss2.str().c_str());
    }
}


template<size_t N, typename T>
void compare_ref_x<N, T>::compare(const char *test, block_tensor_rd_i<N, T> &t,
    block_tensor_rd_i<N, T> &t_ref, T thresh) {

    bto_compare<N, T> cmp(t, t_ref, thresh);
    if(!cmp.compare()) {
        std::ostringstream str;
        str << "In " << test << ": ";
        cmp.tostr(str);
        /*
        str << "Result does not match reference ";
        if ( ! cmp.get_diff().m_number_of_orbits )
            str << "symmetry";
        else if ( ! cmp.get_diff().m_similar_orbit ) {
            str << "symmetry: Orbits with canonical blocks "
                << cmp.get_diff().m_canonical_block_index_1 << " (act) vs. "
                << cmp.get_diff().m_canonical_block_index_2 << " (ref) differ.";
        }
        else if ( cmp.get_diff().m_zero_1 != cmp.get_diff().m_zero_2 ) {
            str << "at zero block "
                << cmp.get_diff().m_canonical_block_index_1
                << ": "
                << (cmp.get_diff().m_zero_1 ? "Z" : "NZ")
                << " (act) vs. "
                << (cmp.get_diff().m_zero_2 ? "Z" : "NZ")
                << " (ref).";
        }
        else {
            str << "in block " << cmp.get_diff().m_canonical_block_index_1
                << " at element "
                << cmp.get_diff().m_inblock << ": "
                << cmp.get_diff().m_diff_elem_1 << " (act) vs. "
                << cmp.get_diff().m_diff_elem_2 << " (ref), "
                << cmp.get_diff().m_diff_elem_1 - cmp.get_diff().m_diff_elem_2
                << " (diff)";
        }*/
        throw_exc("compare_ref_x", "compare()", str.str().c_str());
    }
}


template<size_t N, typename T>
void compare_ref_x<N, T>::compare(const char *test, const symmetry<N, T> &s,
        const symmetry<N, T> &s_ref) {

    if(!s_ref.get_bis().equals(s.get_bis())) {
        std::ostringstream ss;
        ss << "In " << test << ": Different block index spaces.";
        throw_exc("compare_ref_x", "compare()", ss.str().c_str());
    }

    orbit_list<N, T> ol(s), ol_ref(s_ref);
    for(typename orbit_list<N, T>::iterator io_ref = ol_ref.begin();
        io_ref != ol_ref.end(); io_ref++) {

        if(!ol.contains(ol_ref.get_abs_index(io_ref))) {
            std::ostringstream ss;
            index<N> idx;
            ol_ref.get_index(io_ref, idx);
            ss << "In " << test << ": Canonical index " << idx
                << " is absent from result.";
            throw_exc("compare_ref_x", "compare()", ss.str().c_str());
        }
    }
    for(typename orbit_list<N, T>::iterator io = ol.begin();
        io != ol.end(); io++) {

        if(!ol_ref.contains(ol.get_abs_index(io))) {
            std::ostringstream ss;
            index<N> idx;
            ol.get_index(io, idx);
            ss << "In " << test << ": Canonical index " << idx
                << " is absent from reference.";
            throw_exc("compare_ref_x", "compare()", ss.str().c_str());
        }
    }
}


template<size_t N, typename T>
void compare_ref_x<N, T>::compare(const char *test,
        const block_index_space<N> &bis,
        const symmetry_element_set<N, T> &s,
        const symmetry_element_set<N, T> &s_ref) {

    symmetry<N, T> sym(bis), sym_ref(bis);
    for(typename symmetry_element_set<N, T>::const_iterator i =
        s.begin(); i != s.end(); i++) sym.insert(s.get_elem(i));
    for(typename symmetry_element_set<N, T>::const_iterator i =
        s_ref.begin(); i != s_ref.end(); i++) {
        sym_ref.insert(s_ref.get_elem(i));
    }

    compare(test, sym, sym_ref);
}

template<size_t N>
using compare_ref = compare_ref_x<N, double>;  

} // namespace libtensor

#endif // LIBTENSOR_COMPARE_REF_H
