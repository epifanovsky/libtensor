#ifndef LIBTENSOR_COMBINE_PART_H
#define LIBTENSOR_COMBINE_PART_H

#include "../se_part.h"
#include "../symmetry_element_set_adapter.h"

namespace libtensor {

/** \brief Combine multiple se_part<N, T> objects in a %symmetry element set
        to one.

    This class takes a %symmetry element set of se_part<N, T> objects and
    combines into a single se_part<N, T>. The resulting object has to produce
    the same orbits and orbit list as the full set.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class combine_part {
public:
    static const char *k_clazz; //!< Class name

    typedef se_part<N, T> se_t;

private:
    typedef symmetry_element_set_adapter<N, T, se_t> adapter_t;

private:
    adapter_t m_set; // Symmetry element set
    dimensions<N> m_pdims; //!< Partition dimensions
    block_index_space<N> m_bis; //!< Block index space

public:
    /** \brief Constructor
     **/
    combine_part(const symmetry_element_set<N, T> &set);

    /** \brief Obtain block index space
     **/
    const block_index_space<N> &get_bis() const { return m_bis; }

    /** \brief Obtain the partition dimensions of the result
     **/
    const dimensions<N> &get_pdims() const { return m_pdims; }

    /** \brief Do the
     **/
    void perform(se_t &elx);

private:
    static const block_index_space<N> &extract_bis(adapter_t &ad);
    static dimensions<N> make_pdims(adapter_t &ad);
};


template<size_t N, typename T>
const char *combine_part<N, T>::k_clazz = "combine_part<N, T>";

template<size_t N, typename T>
combine_part<N, T>::combine_part(const symmetry_element_set<N, T> &set) :
m_set(set), m_bis(extract_bis(m_set)), m_pdims(make_pdims(m_set)) {
}

template<size_t N, typename T>
void combine_part<N, T>::perform(se_t &el) {

    static const char *method = "perform(se_t &)";
    if (! m_pdims.equals(el.get_pdims())) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "pdims");
    }
    if (! m_bis.equals(el.get_bis())) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "bis");
    }


    // Loop over all partitions in the result se_part
    abs_index<N> ai(m_pdims);
    do {

        const index<N> &i1 = ai.get_index();

        // Loop over all se_part<N, T> in the set
        for (typename adapter_t::iterator it = m_set.begin();
                it != m_set.end(); it++) {

            const se_t &elx = m_set.get_elem(it);
            const dimensions<N> &pdims = elx.get_pdims();

            // Get the respective partition index in current se_part<N, T>
            index<N> ix1;
            for (register size_t i = 0; i < N; i++) {
                if (pdims[i] != 1) { ix1[i] = i1[i]; }
            }

            // Partition is forbidden => result partition is also forbidden
            if (elx.is_forbidden(ix1)) {
                el.mark_forbidden(i1);
                continue;
            }

            // Map is only of interest if ix1 < ix2
            index<N> ix2 = elx.get_direct_map(ix1);
            if (! (ix1 < ix2)) continue;

            bool sx = elx.get_sign(ix1, ix2);
            for (size_t i = 0; i < N; i++) {
                if (pdims[i] == 1) { ix2[i] = i1[i]; }
            }

            if (i1 == ix2) continue;

            // If result partition is already forbidden, mark also the target
            // partition forbidden
            if (el.is_forbidden(i1)) {
                el.mark_forbidden(ix2);
                continue;
            }

            if (el.map_exists(i1, ix2)) {
                bool sign = el.get_sign(i1, ix2);
                if (sx != sign) {
                    el.mark_forbidden(i1);
                }
            }
            else {
                el.add_map(i1, ix2, sx);
            }
        }

    } while (ai.inc());
}

template<size_t N, typename T>
const block_index_space<N> &combine_part<N, T>::extract_bis(adapter_t &ad) {

    if (ad.is_empty()) {
        throw bad_symmetry(g_ns, k_clazz, "extract_bis(adapter_t &)",
                __FILE__, __LINE__, "Empty set.");
    }

    typename adapter_t::iterator it = ad.begin();
    const block_index_space<N> &bis = ad.get_elem(it).get_bis();
    it++;
    for (; it != ad.end(); it++) {

        if (! bis.equals(ad.get_elem(it).get_bis())) {
            throw bad_symmetry(g_ns, k_clazz, "extract_bis(adapter_t &)",
                    __FILE__, __LINE__, "bis");
        }
    }

    return bis;
}

template<size_t N, typename T>
dimensions<N> combine_part<N, T>::make_pdims(adapter_t &ad) {

    if (ad.is_empty()) {
        throw bad_symmetry(g_ns, k_clazz, "make_pdims(adapter_t &)",
                __FILE__, __LINE__, "Empty set.");
    }

    index<N> i1, i2;
    for (typename adapter_t::iterator it = ad.begin(); it != ad.end(); it++) {

        const se_t &el = ad.get_elem(it);
        const dimensions<N> &pdims = el.get_pdims();

        for (size_t i = 0; i < N; i++) {
            if (pdims[i] != 1) {
                if ((i2[i] != 0) && (pdims[i] - 1 != i2[i])) {
                    throw bad_symmetry(g_ns, k_clazz,
                            "make_pdims(adapter_t &)", __FILE__,
                            __LINE__, "pdims");
                }
                i2[i] = pdims[i] - 1;
            }
        }
    }

    return dimensions<N>(index_range<N>(i1, i2));
}

} // namespace libtensor

#endif // LIBTENSOR_COMBINE_PART_H
