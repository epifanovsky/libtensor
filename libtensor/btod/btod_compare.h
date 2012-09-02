#ifndef LIBTENSOR_BTOD_COMPARE_H
#define LIBTENSOR_BTOD_COMPARE_H

#include <cmath> // for fabs
#include <sstream>
#include "../defs.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/transf_list.h"
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_compare.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "bad_block_index_space.h"

namespace libtensor {


/** \brief Compares two block tensors
    \tparam N Tensor order.

    This operation takes two block tensors with the same block %index space,
    compares them, and returns a structure that contains the first
    difference found.

    The constructor takes two block tensors, compare() performs the actual
    comparison and returns true if the block tensors are identical, false
    otherwise. When compare() returns false, the difference structure is
    available from get_diff().

    Along with two block tensors, the constructor takes the threshold for
    comparing data (the elements of the block tensors may not differ by
    more than the absolute value of the threshold) as well as a boolean
    parameter that enables or disables strict zero block comparison policy.

    When the strict zero block policy is on, the blocks marked as zero
    blocks in one of the block tensors must also be marked zero in the other
    one. When the policy is off, and a block is marked zero in one of the
    block tensors, it can be either marked zero in the other one or contain
    all zero elements within the threshold.

    The symmetries of the compared block tensors must yield identical
    orbits.

    Upon return from compare(), if differences are found, a structure is
    filled with data describing them.

    The main element of the difference structure is the kind of difference.
    Possible values are:
     - diff::DIFF_NODIFF - No differences found. Other elements of the
        structure have default values and are meaningless.
     - diff::DIFF_ORBLSTSZ - Orbit lists have different sizes. Further
        comparison is not performed, other members of the structure are
        meaningless.
     - diff::DIFF_ORBIT - Difference found in orbit lists. \c bidx
        identifies the block %index where the difference was found, and
        \c can1 and \c can2 contain whether the block is canonical.
     - diff::DIFF_DATA - Difference found in the canonical block data.
        \c bidx identifies the block, \c idx reads the position at which
        the difference is found. \c zero1 and \c zero2 specify whether
        the block is zero in the two tensors. \c data1 and \c data2 give
        the values that are different (only when \c zero1=false and
        \c zero2=false).

    Two special static methods tostr() will output the difference structure
    to a stream or a string in a human-readable format.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_compare {
public:
    static const char *k_clazz; //!< Class name

public:
    struct diff {
        enum {
            DIFF_NODIFF, //!< No differences found
            DIFF_ORBLSTSZ, //!< Different orbit list sizes
            DIFF_ORBIT, //!< Different orbits
            DIFF_TRANSF, //!< Different transformation within orbit
            DIFF_DATA //!< Difference in data
        };

        unsigned kind;
        index<N> bidx;
        index<N> idx;
        bool can1, can2;
        bool zero1, zero2;
        double data1, data2;
    };

private:

    block_tensor_i<N, double> &m_bt1; //!< First block %tensor
    block_tensor_i<N, double> &m_bt2; //!< Second block %tensor
    double m_thresh; //!< Threshold
    bool m_strict; //!< Strict zero blocks
    diff m_diff; //!< Difference structure

public:
    /** \brief Initializes the operation
        \param bt1 First %tensor.
        \param bt2 Second %tensor.
        \param thresh Equality threshold.
        \param strict Strict check of zero blocks.

        The two block tensors must have compatible block index spaces,
        otherwise an exception will be thrown.
     **/
    btod_compare(block_tensor_i<N, double> &bt1,
        block_tensor_i<N, double> &bt2, double thresh = 0.0,
        bool strict = true);

    /** \brief Performs the comparison
        \return \c true if all the elements are equal within
            the threshold, \c false otherwise
     **/
    bool compare();

    /** \brief Returns the difference structure
     **/
    const diff &get_diff() const {

        return m_diff;
    }

    /** \brief Prints the contents of the difference structure to
            a stream in a human-readable form
     **/
    void tostr(std::ostream &s);

    /** \brief Appends the contents of the difference structure in
            a human-readable form to the end of the string
     **/
    void tostr(std::string &s);

private:
    /** \brief Checks that two orbits have the same canonical %index
     **/
    bool compare_canonical(const abs_index<N> &acidx1, orbit<N, double> &o1,
        orbit<N, double> &o2);

    /** \brief Checks that the same transformation corresponds to a
            given %index
     **/
    bool compare_transf(const abs_index<N> &aidx, orbit<N, double> &o1,
        transf_list<N, double> &trl1, orbit<N, double> &o2,
        transf_list<N, double> &trl2);

    /** \brief Compares two canonical blocks identified by an %index
     **/
    bool compare_data(const abs_index<N> &aidx,
        block_tensor_ctrl<N, double> &ctrl1,
        block_tensor_ctrl<N, double> &ctrl2);

    /** \brief Checks whether the %tensor is filled with zeros,
            within the threshold
     **/
    bool check_zero(dense_tensor_i<N, double> &t);

private:
    btod_compare(const btod_compare<N>&);
    btod_compare<N> &operator=(const btod_compare<N>&);
};


template<size_t N>
const char *btod_compare<N>::k_clazz = "btod_compare<N>";


template<size_t N>
btod_compare<N>::btod_compare(block_tensor_i<N, double> &bt1,
    block_tensor_i<N, double> &bt2, double thresh, bool strict) :

    m_bt1(bt1), m_bt2(bt2), m_thresh(fabs(thresh)), m_strict(strict) {

    static const char *method = "btod_compare(block_tensor_i<N, double>&, "
        "block_tensor_i<N, double>&, double, bool)";

    if(!m_bt1.get_bis().equals(m_bt2.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bt1, bt2");
    }
}


template<size_t N>
bool btod_compare<N>::compare() {

    //
    //  Start with the assumption that the block tensor are identical
    //
    index<N> i0;
    m_diff.kind = diff::DIFF_NODIFF;
    m_diff.bidx = i0;
    m_diff.can1 = true;
    m_diff.can2 = true;
    m_diff.zero1 = true;
    m_diff.zero2 = true;
    m_diff.idx = i0;
    m_diff.data1 = 0.0;
    m_diff.data2 = 0.0;
    if(&m_bt1 == &m_bt2) return true;

    block_tensor_ctrl<N, double> ctrl1(m_bt1), ctrl2(m_bt2);
    orbit_list<N, double> ol1(ctrl1.req_const_symmetry()),
        ol2(ctrl2.req_const_symmetry());
    dimensions<N> bidims = m_bt1.get_bis().get_block_index_dims();

    //
    //  If orbit lists are different
    //
    if(ol1.get_size() != ol2.get_size()) {

        m_diff.kind = diff::DIFF_ORBLSTSZ;
        return false;
    }

    //
    //  Compare all canonical indexes
    //
    for(typename orbit_list<N, double>::iterator io1 = ol1.begin();
        io1 != ol1.end(); io1++) {

        if(!ol2.contains(ol1.get_abs_index(io1))) {

            m_diff.kind = diff::DIFF_ORBIT;
            m_diff.bidx = ol1.get_index(io1);
            m_diff.can1 = true;
            m_diff.can2 = false;
            return false;
        }
    }

    //
    //  Compare orbits
    //
    for(typename orbit_list<N, double>::iterator io1 = ol1.begin();
        io1 != ol1.end(); io1++) {

        orbit<N, double> o1(ctrl1.req_const_symmetry(),
            ol1.get_index(io1));

        for(typename orbit<N, double>::iterator i1 = o1.begin();
            i1 != o1.end(); i1++) {

            abs_index<N> ai1(o1.get_abs_index(i1), bidims);
            orbit<N, double> o2(ctrl2.req_const_symmetry(),
                ai1.get_index());
            transf_list<N, double> trl1(ctrl1.req_const_symmetry(),
                ai1.get_index());
            transf_list<N, double> trl2(ctrl2.req_const_symmetry(),
                ai1.get_index());

            if(!compare_canonical(ai1, o1, o2)) return false;
            if(!compare_transf(ai1, o1, trl1, o2, trl2))
                return false;
        }
    }

    //
    //  Compare actual data
    //
    for(typename orbit_list<N, double>::iterator io1 = ol1.begin();
        io1 != ol1.end(); io1++) {

        abs_index<N> ai(ol1.get_index(io1), bidims);
        if(!compare_data(ai, ctrl1, ctrl2)) return false;
    }

    return true;
}


template<size_t N>
bool btod_compare<N>::compare_canonical(const abs_index<N> &acidx1,
    orbit<N, double> &o1, orbit<N, double> &o2) {

    if(o1.get_abs_canonical_index() != o2.get_abs_canonical_index()) {

        m_diff.kind = diff::DIFF_ORBIT;
        m_diff.bidx = acidx1.get_index();
        m_diff.can1 = true;
        m_diff.can2 = false;
        return false;
    }

    return true;
}


template<size_t N>
bool btod_compare<N>::compare_transf(const abs_index<N> &aidx,
    orbit<N, double> &o1, transf_list<N, double> &trl1,
    orbit<N, double> &o2, transf_list<N, double> &trl2) {

    bool diff = false;
    for(typename transf_list<N, double>::iterator i = trl1.begin();
        !diff && i != trl1.end(); i++) {
        diff = !trl2.is_found(trl1.get_transf(i));
    }
    for(typename transf_list<N, double>::iterator i = trl2.begin();
        !diff && i != trl2.end(); i++) {
        diff = !trl1.is_found(trl2.get_transf(i));
    }

    if(diff) {

        m_diff.kind = diff::DIFF_TRANSF;
        m_diff.bidx = aidx.get_index();
        m_diff.can1 =
            aidx.get_abs_index() == o1.get_abs_canonical_index();
        m_diff.can2 =
            aidx.get_abs_index() == o2.get_abs_canonical_index();
        return false;
    }

    return true;
}


template<size_t N>
bool btod_compare<N>::compare_data(const abs_index<N> &aidx,
    block_tensor_ctrl<N, double> &ctrl1,
    block_tensor_ctrl<N, double> &ctrl2) {

    bool zero1 = ctrl1.req_is_zero_block(aidx.get_index());
    bool zero2 = ctrl2.req_is_zero_block(aidx.get_index());

    if(zero1 != zero2) {

        if(m_strict) {

            m_diff.kind = diff::DIFF_DATA;
            m_diff.bidx = aidx.get_index();
            m_diff.zero1 = zero1;
            m_diff.zero2 = zero2;
            return false;
        } else {

            block_tensor_ctrl<N, double> &c = zero2 ? ctrl1 : ctrl2;
            index<N> idx;
            bool z = check_zero(c.req_block(aidx.get_index()));
            c.ret_block(aidx.get_index());
            if(!z) {
                m_diff.kind = diff::DIFF_DATA;
                m_diff.bidx = aidx.get_index();
                m_diff.zero1 = false;
                m_diff.zero2 = false;
                if(zero1) {
                    m_diff.data2 = m_diff.data1;
                    m_diff.data1 = 0.0;
                } else {
                    m_diff.data2 = 0.0;
                }
                return false;
            }
        }

        return true;
    }

    if(zero1) return true;

    dense_tensor_i<N, double> &t1 = ctrl1.req_block(aidx.get_index());
    dense_tensor_i<N, double> &t2 = ctrl2.req_block(aidx.get_index());

    tod_compare<N> cmp(t1, t2, m_thresh);
    if(cmp.compare()) return true;

    m_diff.kind = diff::DIFF_DATA;
    m_diff.bidx = aidx.get_index();
    m_diff.idx = cmp.get_diff_index();
    m_diff.can1 = true;
    m_diff.can2 = true;
    m_diff.zero1 = false;
    m_diff.zero2 = false;
    m_diff.data1 = cmp.get_diff_elem_1();
    m_diff.data2 = cmp.get_diff_elem_2();

    return false;
}


template<size_t N>
bool btod_compare<N>::check_zero(dense_tensor_i<N, double> &t) {

    dense_tensor_ctrl<N, double> c(t);
    const double *p = c.req_const_dataptr();
    size_t sz = t.get_dims().get_size();
    bool ok = true;
    for(size_t i = 0; i < sz; i++) {
        if(fabs(p[i]) > m_thresh) {
            ok = false;
            abs_index<N> aidx(i, t.get_dims());
            m_diff.idx = aidx.get_index();
            m_diff.data1 = p[i];
            break;
        }
    }
    c.ret_const_dataptr(p);
    return ok;
}


template<size_t N>
void btod_compare<N>::tostr(std::ostream &s) {

    if(m_diff.kind == diff::DIFF_NODIFF) {
        s << "No differences found.";
        return;
    }

    if(m_diff.kind == diff::DIFF_ORBLSTSZ) {
        s << "Different number of orbits.";
        return;
    }

    if(m_diff.kind == diff::DIFF_ORBIT) {
        s << "Different orbits at block " << m_diff.bidx << " "
            << (m_diff.can1 ? "canonical" : "not canonical")
            << " (1), "
            << (m_diff.can2 ? "canonical" : "not canonical")
            << " (2).";
        return;
    }

    if(m_diff.kind == diff::DIFF_TRANSF) {
        s << "Different transformations for block " << m_diff.bidx
            << ".";
        return;
    }

    if(m_diff.kind == diff::DIFF_DATA) {

        if(m_diff.zero1 != m_diff.zero2) {

            s << "Difference found at zero block " << m_diff.bidx
                << " "
                << (m_diff.zero1 ? "zero" : "not zero")
                << " (1), "
                << (m_diff.zero2 ? "zero" : "not zero")
                << " (2).";
        } else {

            s << "Difference found at block " << m_diff.bidx
                << ", element " << m_diff.idx
                << " " << m_diff.data1 << " (1), "
                << m_diff.data2 << " (2), "
                << m_diff.data2 - m_diff.data1 << " (diff).";
        }
        return;
    }

    s << "Difference found.";
}


template<size_t N>
void btod_compare<N>::tostr(std::string &s) {

    std::ostringstream ss;
    tostr(ss);
    s += ss.str();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COMPARE_H
