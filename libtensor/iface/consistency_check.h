#ifndef LIBTENSOR_CONSISTENCY_CHECK_H
#define LIBTENSOR_CONSISTENCY_CHECK_H

#include "../core/symmetry_element_set.h"
#include "bad_symmetry.h"
#include "se_label.h"
#include "se_part.h"
#include "se_perm.h"

namespace libtensor {

/** \brief Consistency check for symmetry element sets
    \tparam ElemT Type of symmetry element
 **/
template<size_t N, typename T, typename ElemT>
class consistency_check {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef ElemT element_t;

private:
    symmetry_element_set_adapter<N, T, ElemT> m_adapter;

public:
    consistency_check(const symmetry_element_set<N, T> &set) :
        m_adapter(set) {

        if (set.get_id() != ElemT::k_sym_type) {
            throw bad_symmetry(g_ns, k_clazz, "consistency_check()",
                    __FILE__, __LINE__, "Unknown symmetry element type.");
        }
    }

    void perform() throw(bad_symmetry) { }
};


/** \brief Specialization of consistency check for se_label
 **/
template<size_t N, typename T>
class consistency_check< N, T, se_label<N, T> > {
public:
    static const char *k_clazz;

public:
    typedef se_label<N, T> element_t;

private:
    symmetry_element_set_adapter<N, T, element_t> m_adapter;

public:
    consistency_check(const symmetry_element_set<N, T> &set) :
        m_adapter(set) {

        if (set.get_id() != element_t::k_sym_type) {
            throw bad_symmetry(g_ns, k_clazz, "consistency_check()",
                    __FILE__, __LINE__, "No se_label elements in set.");
        }
    }

    void perform() throw(bad_symmetry);
};

/** \brief Specialization of consistency check for se_part

 **/
template<size_t N, typename T>
class consistency_check< N, T, se_part<N, T> > {
public:
    static const char *k_clazz;

public:
    typedef se_part<N, T> element_t;

private:
    symmetry_element_set_adapter<N, T, element_t> m_adapter;

public:
    consistency_check(const symmetry_element_set<N, T> &set) :
        m_adapter(set) {

        if (set.get_id() != element_t::k_sym_type) {
            throw bad_symmetry(g_ns, k_clazz, "consistency_check()",
                    __FILE__, __LINE__, "No se_part elements in set.");
        }
    }

    void perform() throw(bad_symmetry);
};

/** \brief Specialization of consistency check for se_perm
 **/
template<size_t N, typename T>
class consistency_check< N, T, se_perm<N, T> > {
public:
    static const char *k_clazz;

public:
    typedef se_perm<N, T> element_t;

private:
    symmetry_element_set_adapter<N, T, element_t> m_adapter;

public:
    consistency_check(const symmetry_element_set<N, T> &set) :
        m_adapter(set) {

        if (set.get_id() != element_t::k_sym_type) {
            throw bad_symmetry(g_ns, k_clazz, "consistency_check()",
                    __FILE__, __LINE__, "No se_perm elements in set.");
        }
    }

    void perform() throw(bad_symmetry);
};

template<size_t N, typename T, typename ElemT>
const char *consistency_check<N, T, ElemT>::k_clazz =
        "consistency_check<N, T, ElemT>";

template<size_t N, typename T>
const char *consistency_check< N, T, se_label<N, T> >::k_clazz =
        "consistency_check< N, T, se_label<N, T> >";

template<size_t N, typename T>
const char *consistency_check< N, T, se_part<N, T> >::k_clazz =
        "consistency_check< N, T, se_part<N, T> >";

template<size_t N, typename T>
const char *consistency_check< N, T, se_perm<N, T> >::k_clazz =
        "consistency_check< N, T, se_perm<N, T> >";

template<size_t N, typename T>
void consistency_check< N, T, se_label<N, T> >::perform() throw(bad_symmetry) {

    try {

    }
    catch (exception &e) {
        throw bad_symmetry(g_ns, k_clazz, "perform()",
                __FILE__, __LINE__, e.what());
    }
}

template<size_t N, typename T>
void consistency_check< N, T, se_part<N, T> >::perform() throw(bad_symmetry) {

    try {

    }
    catch (exception &e) {
        throw bad_symmetry(g_ns, k_clazz, "perform()",
                __FILE__, __LINE__, e.what());
    }
}

template<size_t N, typename T>
void consistency_check< N, T, se_perm<N, T> >::perform() throw(bad_symmetry) {

    try {

    permutation_group<N, T> g(m_set);

    }
    catch (exception &e) {
        throw bad_symmetry(g_ns, k_clazz, "perform()",
                __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor

#endif // LIBTENSOR_CONSISTENCY_CHECK_H
