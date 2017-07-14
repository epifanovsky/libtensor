#include <typeinfo>
#include <libtensor/core/symmetry_element_set.h>
#include "../test_utils.h"

using namespace libtensor;


namespace {

template<size_t N>
class sym_elem_1 : public symmetry_element_i<N, double> {
private:
    static size_t m_count;
    size_t m_m, m_n;

public:
    sym_elem_1(size_t m) : m_m(m), m_n(0) { m_count++; }
    virtual ~sym_elem_1() { m_count--; }
    virtual const char *get_type() const { return "sym_elem_1"; }
    virtual const mask<N> &get_mask() const { throw 0; }
    virtual void permute(const permutation<N> &perm) { throw 0; }
    virtual bool is_valid_bis(const block_index_space<N> &bis) const {
        throw 0;
    }
    virtual bool is_allowed(const index<N> &idx) const { throw 0; }
    virtual void apply(index<N> &idx) const { throw 0; }
    virtual void apply(index<N> &idx, tensor_transf<N, double> &tr) const {
        throw 0;
    }
    virtual symmetry_element_i<N, double> *clone() const {
        return new sym_elem_1(m_m, m_n + 1);
    }
    size_t get_m() const { return m_m; }
    size_t get_n() const { return m_n; }
    static size_t get_count() { return m_count; }

private:
    sym_elem_1(size_t m, size_t n) : m_m(m), m_n(n) { m_count++; }
};

template<size_t N>
size_t sym_elem_1<N>::m_count = 0;

}


/** \test Tests the construction and iterators on the empty set
 **/
int test_1() {

    static const char testname[] = "symmetry_element_set_test::test_1()";

    try {

    symmetry_element_set<2, double> set("sym_elem_1");
    if(!set.is_empty()) {
        return fail_test(testname, __FILE__, __LINE__,
            "!set.is_empty() in empty set.");
    }
    symmetry_element_set<2, double>::iterator i1 = set.begin();
    if(i1 != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() != set.end() in empty set.");
    }
    symmetry_element_set<2, double>::const_iterator i2 = set.begin();
    if(i2 != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() != set.end() in empty set (const).");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests the addition and life time of one element
 **/
int test_2() {

    static const char testname[] = "symmetry_element_set_test::test_2()";

    try {

    sym_elem_1<2> e1(1); // This object has n=0, its clones have n!=0
    symmetry_element_set<2, double> set("sym_elem_1");
    set.insert(e1);

    //  One instance is local (e1) + one inside set
    if(sym_elem_1<2>::get_count() != 2) {
        return fail_test(testname, __FILE__, __LINE__,
            "sym_elem_1<2>::get_count() != 2 (1).");
    }

    //  Test the validity of the iterator

    symmetry_element_set<2, double>::iterator i1 = set.begin();
    if(i1 == set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() == set.end() in non-empty set.");
    }

    //  Check the element type

    try {
        sym_elem_1<2> &e1a = dynamic_cast< sym_elem_1<2>& >(
            set.get_elem(i1));
    } catch(std::bad_cast&) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set has the wrong type.");
    }

    //  Check that the element in the set equals the original one

    sym_elem_1<2> &e1a = dynamic_cast< sym_elem_1<2>& >(set.get_elem(i1));
    if(e1a.get_m() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set is incorrectly initialized.");
    }

    //  Make sure that the element was properly cloned

    if(e1a.get_n() == 0) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set is not cloned properly.");
    }

    //  Check that there is only one element in the set
    i1++;
    if(i1 != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "The set contains more than one element.");
    }

    //  Remove the element from the set

    i1 = set.begin();
    set.remove(i1);
    if(!set.is_empty()) {
        return fail_test(testname, __FILE__, __LINE__,
            "!set.is_empty() in empty set.");
    }
    if(set.begin() != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() != set.end() in empty set.");
    }

    //  Check for memory leaks. Only one local instance should exist

    if(sym_elem_1<2>::get_count() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "sym_elem_1<2>::get_count() != 1 (1).");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests the addition and life time of one element
        (use const iterators)
 **/
int test_3() {

    static const char testname[] = "symmetry_element_set_test::test_3()";

    try {

    sym_elem_1<2> e1(1); // This object has n=0, its clones have n!=0
    symmetry_element_set<2, double> set("sym_elem_1");
    set.insert(e1);

    //  One instance is local (e1) + one inside set
    if(sym_elem_1<2>::get_count() != 2) {
        return fail_test(testname, __FILE__, __LINE__,
            "sym_elem_1<2>::get_count() != 2 (1).");
    }

    //  Test the validity of the iterator

    symmetry_element_set<2, double>::const_iterator i1 = set.begin();
    if(i1 == set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() == set.end() in non-empty set.");
    }

    //  Check the element type

    try {
        const sym_elem_1<2> &e1a = dynamic_cast< const sym_elem_1<2>& >(
            set.get_elem(i1));
    } catch(std::bad_cast&) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set has the wrong type.");
    }

    //  Check that the element in the set equals the original one

    const sym_elem_1<2> &e1a =
        dynamic_cast< const sym_elem_1<2>& >(set.get_elem(i1));
    if(e1a.get_m() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set is incorrectly initialized.");
    }

    //  Make sure that the element was properly cloned

    if(e1a.get_n() == 0) {
        return fail_test(testname, __FILE__, __LINE__,
            "Element in the set is not cloned properly.");
    }

    //  Check that there is only one element in the set
    i1++;
    if(i1 != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "The set contains more than one element.");
    }

    //  Remove all elements and check the iterator
    set.clear();
    if(!set.is_empty()) {
        return fail_test(testname, __FILE__, __LINE__,
            "!set.is_empty() in empty set.");
    }
    if(set.begin() != set.end()) {
        return fail_test(testname, __FILE__, __LINE__,
            "set.begin() != set.end() in empty set.");
    }

    //  Check for memory leaks. Only one local instance should exist

    if(sym_elem_1<2>::get_count() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "sym_elem_1<2>::get_count() != 1 (1).");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_1() |
    test_2() |
    test_3() |

    0;
}

