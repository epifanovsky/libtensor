#ifndef LIBTENSOR_GEN_BTO_COMPARE_IMPL_H
#define LIBTENSOR_GEN_BTO_COMPARE_IMPL_H

#include <libtensor/core/orbit_list.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../gen_bto_compare.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *gen_bto_compare<N, Traits>::k_clazz = "gen_bto_compare<N, Traits>";


template<size_t N, typename Traits>
gen_bto_compare<N, Traits>::gen_bto_compare(
        gen_block_tensor_rd_i<N, bti_traits> &bt1,
        gen_block_tensor_rd_i<N, bti_traits> &bt2,
        const element_type &thresh, bool strict) :

    m_bt1(bt1), m_bt2(bt2), m_thresh(thresh), m_strict(strict) {

    static const char *method = "gen_bto_compare("
            "gen_block_tensor_rd_i<N, bti_traits>&, "
            "gen_block_tensor_rd_i<N, bti_traits>&, "
            "const element_type &, bool)";

    if(!m_bt1.get_bis().equals(m_bt2.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method,
            __FILE__, __LINE__, "bt1, bt2");
    }
}


template<size_t N, typename Traits>
bool gen_bto_compare<N, Traits>::compare() {

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
    m_diff.data1 = Traits::zero();
    m_diff.data2 = Traits::zero();
    if(&m_bt1 == &m_bt2) return true;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl1(m_bt1), ctrl2(m_bt2);
    orbit_list<N, element_type> ol1(ctrl1.req_const_symmetry()),
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
    for(typename orbit_list<N, element_type>::iterator io1 = ol1.begin();
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
    for(typename orbit_list<N, element_type>::iterator io1 = ol1.begin();
        io1 != ol1.end(); io1++) {

        orbit<N, element_type> o1(ctrl1.req_const_symmetry(),
            ol1.get_index(io1));

        for(typename orbit<N, element_type>::iterator i1 = o1.begin();
            i1 != o1.end(); i1++) {

            abs_index<N> ai1(o1.get_abs_index(i1), bidims);
            orbit<N, element_type> o2(ctrl2.req_const_symmetry(),
                ai1.get_index());
            transf_list<N, element_type> trl1(ctrl1.req_const_symmetry(),
                ai1.get_index());
            transf_list<N, element_type> trl2(ctrl2.req_const_symmetry(),
                ai1.get_index());

            if(!compare_canonical(ai1, o1, o2)) return false;
            if(!compare_transf(ai1, o1, trl1, o2, trl2))
                return false;
        }
    }

    //
    //  Compare actual data
    //
    for(typename orbit_list<N, element_type>::iterator io1 = ol1.begin();
        io1 != ol1.end(); io1++) {

        abs_index<N> ai(ol1.get_index(io1), bidims);
        if(!compare_data(ai, ctrl1, ctrl2)) return false;
    }

    return true;
}


template<size_t N, typename Traits>
bool gen_bto_compare<N, Traits>::compare_canonical(const abs_index<N> &acidx1,
    orbit<N, element_type> &o1, orbit<N, element_type> &o2) {

    if(o1.get_abs_canonical_index() != o2.get_abs_canonical_index()) {

        m_diff.kind = diff::DIFF_ORBIT;
        m_diff.bidx = acidx1.get_index();
        m_diff.can1 = true;
        m_diff.can2 = false;
        return false;
    }

    return true;
}


template<size_t N, typename Traits>
bool gen_bto_compare<N, Traits>::compare_transf(const abs_index<N> &aidx,
    orbit<N, element_type> &o1, transf_list<N, element_type> &trl1,
    orbit<N, element_type> &o2, transf_list<N, element_type> &trl2) {

    bool diff = false;
    for(typename transf_list<N, element_type>::iterator i = trl1.begin();
        !diff && i != trl1.end(); i++) {
        diff = !trl2.is_found(trl1.get_transf(i));
    }
    for(typename transf_list<N, element_type>::iterator i = trl2.begin();
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


template<size_t N, typename Traits>
bool gen_bto_compare<N, Traits>::compare_data(const abs_index<N> &aidx,
    gen_block_tensor_rd_ctrl<N, bti_traits> &ctrl1,
    gen_block_tensor_rd_ctrl<N, bti_traits> &ctrl2) {

    typedef typename Traits::template to_compare_type<N>::type to_compare;
    typedef typename Traits::template temp_block_tensor_type<N>::type
            temp_block_tensor_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;


    const index<N> &idx = aidx.get_index();
    bool zero1 = ctrl1.req_is_zero_block(idx);
    bool zero2 = ctrl2.req_is_zero_block(idx);

    if(zero1 != zero2) {

        if(m_strict) {

            m_diff.kind = diff::DIFF_DATA;
            m_diff.bidx = idx;
            m_diff.zero1 = zero1;
            m_diff.zero2 = zero2;
            return false;
        } else {

            gen_block_tensor_rd_ctrl<N, bti_traits> &ca =
                    (zero2 ? ctrl1 : ctrl2);
            rd_block_type &blka = ca.req_const_block(idx);

            bool z;

            temp_block_tensor_type btc(m_bt1.get_bis());

            {
            gen_block_tensor_rd_ctrl<N, bti_traits> cb(btc);
            rd_block_type &blkb = cb.req_const_block(aidx.get_index());

            to_compare cmp(blka, blkb, m_thresh);
            z = cmp.compare();

            ca.ret_const_block(idx);
            cb.ret_const_block(idx);
            }

            {
            gen_block_tensor_wr_ctrl<N, bti_traits> cb(btc);
            cb.req_zero_block(idx);
            }

            if(!z) {
                m_diff.kind = diff::DIFF_DATA;
                m_diff.bidx = idx;
                m_diff.zero1 = false;
                m_diff.zero2 = false;
                if(zero1) {
                    m_diff.data2 = m_diff.data1;
                    m_diff.data1 = Traits::zero();
                } else {
                    m_diff.data2 = Traits::zero();
                }
                return false;
            }
        }

        return true;
    }

    if(zero1) return true;

    rd_block_type &t1 = ctrl1.req_const_block(idx);
    rd_block_type &t2 = ctrl2.req_const_block(idx);

    to_compare cmp(t1, t2, m_thresh);
    if(cmp.compare()) return true;

    m_diff.kind = diff::DIFF_DATA;
    m_diff.bidx = idx;
    m_diff.idx = cmp.get_diff_index();
    m_diff.can1 = true;
    m_diff.can2 = true;
    m_diff.zero1 = false;
    m_diff.zero2 = false;
    m_diff.data1 = cmp.get_diff_elem_1();
    m_diff.data2 = cmp.get_diff_elem_2();

    return false;
}


template<size_t N, typename Traits>
void gen_bto_compare<N, Traits>::tostr(std::ostream &s) {

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


template<size_t N, typename Traits>
void gen_bto_compare<N, Traits>::tostr(std::string &s) {

    std::ostringstream ss;
    tostr(ss);
    s += ss.str();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COMPARE_IMPL_H
